import argparse
from functools import partial
from os.path import join

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import collate_fn, build_vocab
from data import Im2LatexDataset
from model_attn_cnn_init_rnn import Im2LatexModel  # check this
from training import Trainer
from make_vocab import make_vocab
import wandb
from datetime import datetime
import os


def main():
    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    parser.add_argument('--path', required=True, help='root of the model')

    # model args
    parser.add_argument(
        "--emb_dim", type=int, default=80, help="Embedding size")
    parser.add_argument(
        "--enc_rnn_h",
        type=int,
        default=256,
        help="The hidden state of the encoder RNN")
    parser.add_argument(
        "--dec_rnn_h",
        type=int,
        default=512,
        help="The hidden state of the decoder RNN")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./sample_data/",
        help="The dataset's dir")
    # training args
    parser.add_argument(
        "--cuda", action='store_true', default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--test_beam_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="Learning Rate Decay Rate")
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Learning Rate Decay Patience")
    parser.add_argument(
        "--clip", type=float, default=5.0, help="The max gradient norm")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f'./checkpoints/',
        help="The dir to save checkpoints")
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="path to checkpoint, you want to start from"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./sample_data/vocab.pkl",
        help="The path to vocab file")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=4,
        help="The frequency to print message")

    # new args
    parser.add_argument(
        "--cnn",
        type=str,
        default='harvard',
        help="cnn model specification")

    parser.add_argument(
        "--attn",
        type=int,
        default=1,
        help="attention type")

    parser.add_argument(
        "--pos_enc",
        type=str,
        default='none',
        help="positional encoding after cnn encoder")

    parser.add_argument(
        "--dec_init",
        type=int,
        default=0,
        help="decoder hidden states initialization")

    parser.add_argument(
        "--max_len",
        dtype=int,
        default=50,
        help="max predicted sequence length"
    )

    wandb.login()

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # Building vocab
    make_vocab(args.data_path)
    vocab = build_vocab(join(args.data_path, 'vocab.pkl'))

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    args.__dict__['device'] = device

    # data loader
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'train'),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        #pin_memory=True if use_cuda else False,
        num_workers=4)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate'),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        #pin_memory=True if use_cuda else False,
        num_workers=4)

    test_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'test'),
        batch_size=args.test_batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        # pin_memory=True if use_cuda else False,
        num_workers=4)

    # construct model
    vocab_size = len(vocab)
    model = Im2LatexModel(vocab_size, args.emb_dim, args.enc_rnn_h,
                          args.dec_rnn_h, args.cnn, args.attn, args.dec_init,
                          args.pos_enc)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True)

    with wandb.init(project='im2latex', config=args):

        epoch = 0
        global_step = 0

        if args.load_from_checkpoint:
            checkpoint = torch.load(args.load_from_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']

        model = model.to(device)
        # init trainer
        trainer = Trainer(optimizer, model, lr_scheduler, train_loader, val_loader, test_loader, args, epoch, global_step)
        # begin training
        trainer.train()

        trainer.test(beam_size=args.test_beam_size)


if __name__ == "__main__":
    main()
