from functools import partial
from os import makedirs
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
from torchtext.data import metrics
from tqdm.auto import tqdm

import wandb
from Levenshtein import distance

from utils import collate_fn, build_vocab
from data import Im2LatexDataset
from model import Im2LatexModel
from make_vocab import make_vocab
from decoding import LatexProducer


@torch.no_grad()
def evaluate(predictor: LatexProducer,
             data_loader: DataLoader,
             args: dict,
             name: str = 'test'):
    """evaluates the model. Returns bleu score on the dataset

    Args:
        predictor (LatexProducer): decoding with beam search
        data_loader (DataLoader): test dataset
        args (dict): arguments
        name (str, optional): name of the test e.g. val or test for wandb. Defaults to 'test'.
    Returns:
        bleu_score: BLEU score of eval set.
    """
    assert len(data_loader.dataset) > 0
    device = args.device
    log = {}
    BLEUs, edit_dists = [], []
    bleu_score, edit_distance = 0, 1
    loader = tqdm(data_loader, total=len(data_loader.dataset))
    precision_, recall_, f1_ = [], [], []

    for img, _, tgt_formulas in loader:

        if tgt_formulas is None or img is None:
            continue
        img = img.to(device)

        # uncomment next line for beam search decoding
        # pred = predictor._bs_decoding(img)[0].split()

        # uncomment next line for greedy decoding
        pred = predictor._greedy_decoding(img)[0].split()
        tgt_formulas = predictor._idx2formulas(tgt_formulas)[0].split()

        if not tgt_formulas:
            continue

        BLEUs.append(metrics.bleu_score([pred], [tgt_formulas]))

        pred = ' '.join(pred) \
            .replace('<unk>', '') \
            .replace('</s>', '') \
            .replace('<s>', '') \
            .replace('<pad>', '').strip()

        tgt_formulas = ' '.join(tgt_formulas) \
            .replace('<unk>', '') \
            .replace('</s>', '') \
            .replace('<s>', '') \
            .replace('<pad>', '').strip()

        # print()
        # print('-------------------')
        # print('PRED', pred)
        # print()
        # print('TGT', tgt_formulas)

        edit_dists.append(distance(pred, tgt_formulas) / len(tgt_formulas))

        tp, fp, fn = 0, 0, 0
        if len(tgt_formulas) == len(pred) and all([g == p for g, p in zip(tgt_formulas, pred)]):
            tp += len(tgt_formulas)
            continue

        for pred_subtoken in pred:
            if pred_subtoken in tgt_formulas:
                tp += 1
            else:
                fp += 1
        for gt_subtoken in tgt_formulas:
            if gt_subtoken not in pred:
                fn += 1

        precision, recall, f1 = 0.0, 0.0, 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
            log[name + '/precision'] = precision
        if tp + fn > 0:
            recall = tp / (tp + fn)
            log[name + '/recall'] = recall
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            log[name + '/f1'] = f1

        precision_.append(precision)
        recall_.append(recall)
        f1_.append(f1)

        loader.set_description('BLEU: %.3f, ED: %.2e, precision: %.2f, recall: %.2f' %
                               (np.mean(BLEUs), np.mean(edit_dists), np.mean(precision_), np.mean(recall_)))

    if len(BLEUs) > 0:
        bleu_score = np.mean(BLEUs)
        log[name + '/bleu'] = bleu_score

    if len(edit_dists) > 0:
        edit_distance = np.mean(edit_dists)
        log[name + '/edit_distance'] = edit_distance

    precision = np.mean(precision_)
    recall = np.mean(recall_)
    f1 = np.mean(f1_)
    wandb.log(log)

    print('\n%s\n%s' % (tgt_formulas, pred))
    print('BLEU: %.2f' % bleu_score)
    print(f'test precision: {precision}, test recall: {recall}, test f1: {f1}')
    return bleu_score, edit_distance, precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')

    # model params
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

    # data
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl", help="The path to vocab file")
    parser.add_argument('-c', '--checkpoint', default='./ckpt/best.ckpt', type=str, help='path to model checkpoint')
    parser.add_argument('-d', '--data_path', default='./sample_data/', type=str, help='path tot dataset dir')
    parser.add_argument("--save_dir", type=str, default="./results", help="The dir to save results")

    # eval params
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of sequence')

    # other params
    parser.add_argument('--cuda', action='store_true', help='Use cuda or not')

    args = parser.parse_args()

    args.__dict__['device'] = 'cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu'

    # когда научимся нормально считать метрики, будем туда что-то сохранять
    # makedirs(args.__dict__['save_dir'], exist_ok=True)

    make_vocab(args.data_path)
    vocab = build_vocab(args.vocab_path)

    test_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'test'),
        # batch_size=args.test_bs,
        batch_size=1,
        collate_fn=partial(collate_fn, vocab.sign2id),
        # pin_memory=True if use_cuda else False,
        num_workers=1)

    # construct model
    vocab_size = len(vocab)
    model = Im2LatexModel(vocab_size, args.emb_dim, args.enc_rnn_h,
                          args.dec_rnn_h, args.cnn, args.attn,
                          args.pos_enc, args.dec_init)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    predictor = LatexProducer(model, vocab, max_len=args.max_len)

    evaluate(predictor, test_loader, args)
