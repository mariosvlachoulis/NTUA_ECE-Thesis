import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import datetime
from utils.dataset import get_data_loader_folder
from utils.utils import prepare_sub_folder, write_2images, write_html, print_params, adjust_learning_rate
from utils.MattingLaplacian import laplacian_loss_grad

def main():
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_name', default=None, help='Directory name to save')
    parser.add_argument('--mode', type=str, default='photorealistic')
    parser.add_argument('--vgg_ckpoint', type=str, default='checkpoints/vgg_normalised.pth')

    # Dataset
    parser.add_argument('--train_content', default='/home/wenlinfeng/Downloads/unlabeled2017', help='Directory to dataset A')
    parser.add_argument('--train_style', default='/home/wenlinfeng/Downloads/unlabeled2017', help='Directory to dataset B')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--new_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--use_lap', type=bool, default=True)

    # Training options
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)

    parser.add_argument('--style_weight', type=float, default=10)
    parser.add_argument('--content_weight', type=float, default=0)
    parser.add_argument('--lap_weight', type=float, default=1500)
    parser.add_argument('--rec_weight', type=float, default=10)
    parser.add_argument('--temporal_weight', type=float, default=60)

    parser.add_argument('--training_iterations', type=int, default=160000)
    parser.add_argument('--fine_tuning_iterations', type=int, default=10000)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument('--resume_iter', type=int, default=-1)

    # Log
    parser.add_argument('--logs_directory', default='logs', help='Directory to log')
    parser.add_argument('--display_size', type=int, default=16)
    parser.add_argument('--image_display_iter', type=int, default=1000)
    parser.add_argument('--image_save_iter', type=int, default=2000)
    parser.add_argument('--model_save_interval', type=int, default=2000)

    args = parser.parse_args()
    args.base_name = args.base_name or datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    total_iterations = args.training_iterations + args.fine_tuning_iterations
    current_iter = -1

    # Logs directory
    logs_directory = os.path.join(args.logs_directory, args.base_name)
    print("Logs directory:", logs_directory)
    checkpoint_dir, image_dir = prepare_sub_folder(logs_directory)

    # Dataset
    batch_size = args.batch_size
    num_workers = args.batch_size
    new_size = args.new_size
    crop_size = args.crop_size
    train_loader_content = get_data_loader_folder(args.train_content, batch_size, new_size, crop_size, crop_size, use_lap=True)
    train_loader_style = get_data_loader_folder(args.train_style, batch_size, new_size, crop_size, crop_size, use_lap=False)

    # Reversible Network
    from models.RevResNet import RevResNet
    num_blocks = [10, 10, 10]
    strides = [1, 2, 2]
    channels = [16, 64, 256]
    in_channels = 3
    mult = 4
    hidden_dim = 16 if args.mode.lower() == "photorealistic" else 64
    sp_steps = 2 if args.mode.lower() == "photorealistic" else 1
    rev_network = RevResNet(nBlocks=num_blocks, nStrides=strides, nChannels=channels, in_channel=in_channels, mult=mult, hidden_dim=hidden_dim, sp_steps=sp_steps)
    rev_network = rev_network.to(device)
    rev_network.train()
    print_params(rev_network)

    # Optimizer
    optimizer = torch.optim.Adam(rev_network.parameters(), lr=args.lr)

    # Transfer module
    from models.cWCT import cWCT
    cwct = cWCT()

    # VGG for style loss
    from models.VGG import VGG19
    vgg_enc = VGG19(args.vgg_ckpoint)
    vgg_enc.to(device)

    # Resume
    if args.resume:
        state_dict = torch.load(os.path.join(checkpoint_dir, "last.pt"))
        rev_network.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        current_iter = args.resume_iter
        print('Resume from %s. Resume iter is %d' % (os.path.join(checkpoint_dir, "last.pt"), args.resume_iter))

    # Loss
    l1_loss = torch.nn.L1Loss()
    temporal_loss = None

    if args.temporal_weight > 0:
        from utils.TemporalLoss import TemporalLoss
        temporal_loss = TemporalLoss()

    # Training
    iter_loader_content, iter_loader_style = iter(train_loader_content), iter(train_loader_style)
    
    while current_iter < total_iterations:
        images_content, images_style = next(iter_loader_content), next(iter_loader_style)
        laplacian_matrices = []

        if args.lap_weight > 0:
            for M in images_content['laplacian_m']:
                indices = torch.from_numpy(np.vstack((M.row, M.col))).long().to(device)
                values = torch.from_numpy(M.data).to(device)
                shape = torch.Size(M.shape)
                laplacian_m = torch.sparse_coo_tensor(indices, values, shape, device=device)
                laplacian_matrices.append(laplacian_m)

        images_content, images_style = images_content['img'].to(device), images_style['img'].to(device)

        # Optimizer
        adjust_learning_rate(optimizer, args.lr, args.lr_decay, current_iter)
        optimizer.zero_grad()

        # Forward inference
        z_content = rev_network(images_content, forward=True)
        z_style = rev_network(images_style, forward=True)

        # Transfer
        z_content_style = cwct.transfer(z_content, z_style)

        # Backward inference
        stylized = rev_network(z_content_style, forward=False)

        # Style loss
        loss_content, loss_style = vgg_enc(images_content, images_style, stylized, n_layer=4, content_weight=args.content_weight)

        # Cycle reconstruction
        if args.rec_weight > 0:
            z_content_style = rev_network(stylized, forward=True)
            z_style_content = cwct.transfer(z_content_style, z_content)
            reconstructed = rev_network(z_style_content, forward=False)
            loss_reconstruction = l1_loss(reconstructed, images_content)
        else:
            loss_reconstruction = 0

        # Matting Laplacian loss
        if args.lap_weight > 0:
            batch_size = stylized.size(0)
            laplacian_losses = []
            gradient_losses = []

            for i in range(batch_size):
                lap_loss, grad = laplacian_loss_grad(stylized[i], laplacian_matrices[i])
                laplacian_losses.append(lap_loss)
                gradient_losses.append(grad)

            gradient_losses = torch.stack(gradient_losses, dim=0)
            gradient_losses = gradient_losses * args.lap_weight
            gradient_losses = gradient_losses.clamp(-0.05, 0.05)
            stylized.backward(gradient_losses, retain_graph=True)

            loss_laplacian = torch.mean(torch.stack(laplacian_losses, dim=0))
        else:
            loss_laplacian = 0

        # Temporal loss
        if args.temporal_weight > 0 and current_iter > args.training_iterations:
            second_frame, forward_flow = temporal_loss.GenerateFakeData(images_content)
            z_content_second_frame = rev_network(second_frame, forward=True)
            z_content_style_second_frame = cwct.transfer(z_content_second_frame, z_content)
            stylized_second_frame = rev_network(z_content_second_frame.clone(), forward=False)

            loss_temporal, fake_styled_second_frame_1 = temporal_loss(stylized, stylized_second_frame, forward_flow)
            loss_temporal_gt, _ = temporal_loss(images_content, second_frame, forward_flow)
        else:
            loss_temporal = 0
            loss_temporal_gt = 0.

        # Total loss
        loss_total = args.content_weight * loss_content + args.style_weight * loss_style + args.rec_weight * loss_reconstruction + args.temporal_weight * loss_temporal

        loss_total.backward()
        nn.utils.clip_grad_norm_(rev_network.parameters(), 5)
        optimizer.step()

        # Dump training stats in log file
        if (current_iter + 1) % 10 == 0:
            message = "Iteration: %08d/%08d  content_loss:%.4f  lap_loss:%.4f  rec_loss:%.4f  style_loss:%.4f  loss_temporal:%.4f  loss_temporal_gt:%.4f" % (
                current_iter + 1, total_iterations,
                args.content_weight * loss_content,
                args.lap_weight * loss_laplacian,
                args.rec_weight * loss_reconstruction,
                args.style_weight * loss_style,
                args.temporal_weight * loss_temporal,
                args.temporal_weight * loss_temporal_gt,
            )
            print(message)
            with open(logs_directory + "/loss.log", "a") as log_file:
                log_file.write('%s\n' % message)

            # Log sample
            if (current_iter + 1) % args.image_save_iter == 0:
                with torch.no_grad():
                    indices_content = torch.randint(low=0, high=len(train_loader_content.dataset), size=[args.display_size])
                    train_display_images_content = torch.stack([train_loader_content.dataset[i]['img'] for i in indices_content])
                    indices_style = torch.randint(low=0, high=len(train_loader_style.dataset), size=[args.display_size])
                    train_display_images_style = torch.stack([train_loader_style.dataset[i]['img'] for i in indices_style])
                    train_image_outputs = rev_network.sample(cwct, train_display_images_content, train_display_images_style, device)
                write_2images(train_image_outputs, args.display_size, image_dir, 'train_%08d' % (current_iter + 1))
                # HTML
                write_html(logs_directory + "/index.html", current_iter + 1, args.image_save_iter, 'images')

            if (current_iter + 1) % args.image_display_iter == 0:
                with torch.no_grad():
                    indices_content = torch.randint(low=0, high=len(train_loader_content.dataset), size=[args.display_size])
                    train_display_images_content = torch.stack([train_loader_content.dataset[i]['img'] for i in indices_content])
                    indices_style = torch.randint(low=0, high=len(train_loader_style.dataset), size=[args.display_size])
                    train_display_images_style = torch.stack([train_loader_style.dataset[i]['img'] for i in indices_style])
                    image_outputs = rev_network.sample(cwct, train_display_images_content, train_display_images_style, device)
                write_2images(image_outputs, args.display_size, image_dir, 'train_current')

            # Save network weights
            if (current_iter + 1) % args.model_save_interval == 0:
                checkpoint_file = os.path.join(checkpoint_dir, 'last.pt')
                torch.save({'state_dict': rev_network.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_file)

            if (current_iter + 1) == args.training_iterations:
                checkpoint_file = os.path.join(checkpoint_dir, 'model_image.pt')
                torch.save({'state_dict': rev_network.state_dict()}, checkpoint_file)
            elif (current_iter + 1) == total_iterations:
                checkpoint_file = os.path.join(checkpoint_dir, 'model_video.pt')
                torch.save({'state_dict': rev_network.state_dict()}, checkpoint_file)

        current_iter += 1

    print("Finishing training. Model save at %s" % checkpoint_dir)

if __name__ == "__main__":
    main()
