from os.path import join

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model import Im2LatexTransformerModel

from make_vocab import PAD_TOKEN, START_TOKEN, END_TOKEN
from beam_search import BeamSearch
from utils import tile
import wandb


class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, test_loader, args, start_epoch=0, global_step=0):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        self.step = global_step
        self.epoch = start_epoch
        self.best_val_loss = 1e18
        self._metric_skip_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN]

    def train_transformer(self):
        assert isinstance(self.model, Im2LatexTransformerModel), 'Use this only for Im2LatexTransformerModel'
        wandb.watch(self.model)
        total_step = len(self.train_loader)
        val_size = len(self.val_loader)
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            for i, (imgs, tgt4training, tgt4cal_loss) in enumerate(self.train_loader):
                preds = self.model(imgs, tgt4training)
                ys = tgt4cal_loss
                self.optimizer.zero_grad()
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=PAD_TOKEN)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if i % self.args.print_freq == 0:
                    wandb.log({"epoch": epoch, "loss": total_loss / self.args.print_freq})
                    print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
                        self.epoch, i, total_step,
                        100 * i / total_step,
                        total_loss / self.args.print_freq
                    ))

            self.model.eval()
            val_loss = 0
            for imgs, tgt4training, tgt4cal_loss in iter(self.val_loader):
                with torch.no_grad():
                    preds = self.model(imgs, tgt4training)
                    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), tgt4cal_loss, ignore_index=PAD_TOKEN)
                val_loss += loss.item()

            val_loss = val_loss / val_size
            wandb.log({"epoch:": epoch, "val_avg_loss": val_loss})
            print("Epoch {}, validation average loss: {:.4f}".format(
                epoch, val_loss
            ))

            checkpoint = {
                'epoch': epoch,
                'global_step': 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict()
            }

            filename = join(self.args.log_dir, "epoch={epoch:02d}-val_loss={val_loss:.4f}.ckpt".format(
                epoch=epoch, val_loss=val_loss))

            torch.save(checkpoint, filename)

            wandb.save(filename)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
            self.lr_scheduler.step(val_loss)

    def train(self):
        wandb.watch(self.model)
        total_step = len(self.train_loader)
        while self.epoch <= self.args.epochs:
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in self.train_loader:
                step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    wandb.log({"epoch": self.epoch, "loss": losses / self.args.print_freq})
                    print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
                        self.epoch, self.step, total_step,
                        100 * self.step / total_step,
                        losses / self.args.print_freq
                    ))
                    losses = 0.0
            # one epoch Finished, calcute val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()

        imgs = imgs.to(self.args.device)
        tgt4training = tgt4training.to(self.args.device)
        tgt4cal_loss = tgt4cal_loss.to(self.args.device)
        logits = self.model(imgs, tgt4training)

        # calculate loss
        loss = self.cal_loss(logits, tgt4cal_loss)
        self.step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                imgs = imgs.to(self.args.device)
                tgt4training = tgt4training.to(self.args.device)
                tgt4cal_loss = tgt4cal_loss.to(self.args.device)

                logits = self.model(imgs, tgt4training)
                loss = self.cal_loss(logits, tgt4cal_loss)
                val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            wandb.log({"epoch:": self.epoch, "val_avg_loss": avg_loss})
            print("Epoch {}, validation average loss: {:.4f}".format(
                self.epoch, avg_loss
            ))

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict()
        }

        filename = join(self.args.log_dir, "epoch={epoch:02d}-val_loss={val_loss:.4f}.ckpt".format(
            epoch=self.epoch, val_loss=avg_loss))

        torch.save(checkpoint, filename)

        wandb.save(filename)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model()
        return avg_loss

    def test(self, beam_size=1):
        self.model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.test_loader:
                imgs = imgs.to(self.args.device)
                # tgt4training = tgt4training.to(self.args.device)
                tgt4cal_loss = tgt4cal_loss.to(self.args.device)
                if beam_size > 1:
                    pred_tokens = self._beam_search_decoding(imgs, beam_size)
                else:
                    pred_tokens = self._greedy_decoding(imgs)
                batch_size = tgt4cal_loss.size(0)
                if pred_tokens.size(0) != batch_size:
                    raise ValueError(f"Wrong batch size for prediction (expected: {batch_size}, actual: {pred_tokens.size(0)})")
                # print('TGT CAL LOSS SIZE:', tgt4cal_loss.size())
                for example, pred in zip(tgt4cal_loss, pred_tokens):
                      gt_seq = [st for st in example if st not in self._metric_skip_tokens]
                      pred_seq = [st for st in pred if st not in self._metric_skip_tokens]

                      if len(gt_seq) == len(pred_seq) and all([g == p for g, p in zip(gt_seq, pred_seq)]):
                          tp += len(gt_seq)
                          continue

                      for pred_subtoken in pred_seq:
                          if pred_subtoken in gt_seq:
                              tp += 1
                          else:
                              fp += 1
                      for gt_subtoken in gt_seq:
                          if gt_subtoken not in pred_seq:
                              fn += 1

            precision, recall, f1 = 0.0, 0.0, 0.0
            if tp + fp > 0:
                precision = tp / (tp + fp)
            if tp + fn > 0:
                recall = tp / (tp + fn)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            wandb.log({"test precision": precision, "test recall": recall, "test f1": f1})
            print(f'test precision: {precision}, test recall: {recall}, test f1: {f1}')

    def _greedy_decoding(self, imgs):
        enc_outs, hiddens = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs, hiddens)

        batch_size = imgs.size(0)
        max_len = self.args.max_len

        # storing decoding results
        formulas_idx = torch.ones(batch_size, max_len, dtype=torch.long,
                                          device=self.args.device) * PAD_TOKEN
        # first decoding step's input
        tgt = torch.ones(batch_size, 1, dtype=torch.long,
                                        device=self.args.device) * START_TOKEN
        for t in range(max_len):
            dec_states, O_t, logit = self.model.step_decoding(
                dec_states, O_t, enc_outs, tgt)
            tgt = torch.argmax(logit, dim=1, keepdim=True)
            formulas_idx[:, t:t + 1] = tgt

        return formulas_idx

    def _beam_search_decoding(self, imgs, beam_size):
        B = imgs.size(0)
        max_len = self.args.max_len
        # use batch_size*beam_size as new Batch
        imgs = tile(imgs, beam_size, dim=0)
        enc_outs, hiddens = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs, hiddens)

        new_B = imgs.size(0)
        # first decoding step's input
        tgt = torch.ones(new_B, 1).long() * START_TOKEN
        beam = BeamSearch(beam_size, B, self.args.device)
        for t in range(max_len):
            tgt = beam.current_predictions.unsqueeze(1)
            dec_states, O_t, probs = self.model.step_decoding(
                dec_states, O_t, enc_outs, tgt)
            log_probs = torch.log(probs)

            beam.advance(log_probs)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin
            if any_beam_is_finished:
                # Reorder states
                h, c = dec_states
                h = h.index_select(0, select_indices)
                c = c.index_select(0, select_indices)
                dec_states = (h, c)
                O_t = O_t.index_select(0, select_indices)
        # get results
        formulas_idx = torch.stack([hyps[1] for hyps in beam.hypotheses],
                                   dim=0)

        return formulas_idx

    def cal_loss(self, logits, targets):
        """args:
            logits: probability distribution return by model
                    [B, MAX_LEN, voc_size]
            targets: target formulas
                    [B, MAX_LEN]
        """
        padding = torch.ones_like(targets) * PAD_TOKEN
        mask = (targets != padding)

        targets = targets.masked_select(mask)
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, logits.size(2))
        ).contiguous().view(-1, logits.size(2))
        logits = torch.log(logits)

        assert logits.size(0) == targets.size(0)

        loss = F.nll_loss(logits, targets)
        return loss

    def save_model(self):
        print("Saving as best model...")
        torch.save(
            self.model.state_dict(),
            join(self.args.log_dir, 'best.pkl')
        )
