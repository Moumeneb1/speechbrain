#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU with ASR-based transfer learning.

We encode input waveforms into features using a model trained on LibriSpeech,
then feed the features into a seq2seq model to map them to semantics.

(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.decoders.seq2seq import  inflate_tensor
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd


# Define training procedure
class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_intents_bos, tokens_intents_bos_lens = batch.tokens_intents_bos
        tokens_slots_bos, tokens_slots_bos_lens = batch.tokens_slots_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # wav2vec forward pass
        wav2vec2_out = self.modules.wav2vec2(wavs)
        # SLU forward pass
        encoder_out = self.hparams.slu_enc(wav2vec2_out)
        
        e_intents_in = self.hparams.output_emb(tokens_intents_bos)
        e_slots_in = self.hparams.output_emb(tokens_slots_bos)

        h_intent, weights_intent = self.hparams.dec_intent(e_intents_in, encoder_out, wav_lens)
        
        #wieght are vectors or shape [batch,number_of_tokens,timesteps]
        weights_inversed =  torch.unsqueeze(torch.mean(weights_intent.permute(0,2,1),dim=-1),dim=-1)
        #weights_inversed is of size [batch,timesteps,1]
        

        #sleep(30)
        # we add for each timestep a value wichi is the attention weight used for decoding the intent 
        h_slots, weights_slots = self.hparams.dec_slots(e_slots_in, encoder_out, wav_lens,context_intent=torch.mean(h_intent,dim=1))
        #print(encoder_out.shape,weights_slots.shape,h_slots.shape,wav_lens,wav2vec2_out.shape)

        # Output layer for seq2seq log-probabilities
        logits_intent = self.hparams.seq_lin(h_intent)

        #print(logits_intent.shape)
        
        logits_slots = self.hparams.seq_lin(h_slots)


        p_seq_intent = self.hparams.log_softmax(logits_intent)
        p_seq_slots = self.hparams.log_softmax(logits_slots)

        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq_intent,p_seq_slots, wav_lens
        else:
            p_tokens_intent, scores_intent, context_intent = self.hparams.beam_searcher_intent(encoder_out, wav_lens)
            #print("intent context is",context_intent.shape)
            p_tokens_slots, scores_intent_slots = self.hparams.beam_searcher_slots(encoder_out, wav_lens,inflate_tensor(context_intent,80,dim=0))

            return p_seq_intent,p_seq_slots, wav_lens, p_tokens_intent, p_tokens_slots

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq_intent,p_seq_slots, wav_lens = predictions
        else:
            p_seq_intent,p_seq_slots, wav_lens, predicted_tokens_intent, predicted_tokens_slots = predictions

        ids = batch.id
        tokens_intents_eos, tokens_intents_eos_lens = batch.tokens_intents_eos
        tokens_intents, tokens_intents_lens = batch.tokens_intents

        tokens_slots_eos, tokens_slots_eos_lens = batch.tokens_slots_eos
        tokens_slots, tokens_slots_lens = batch.tokens_slots


        loss_seq_intent = self.hparams.seq_cost(
            p_seq_intent, tokens_intents_eos, length=tokens_intents_eos_lens
        )

        loss_seq_slots = self.hparams.seq_cost(
            p_seq_slots, tokens_slots_eos, length=tokens_slots_eos_lens
        )

        loss = 0.5*loss_seq_slots + 0.5*loss_seq_intent

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_intents = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens_intent
            ]

            predicted_slots = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens_slots
            ]

            target_intents = [wrd.split(" ") for wrd in batch.intents]
            target_slots = [wrd.split(" ") for wrd in batch.slots]

            target_semantics =[]
            predicted_semantics = []

            for i in range(len(target_intents)):
                print("aaaaaaaaaaaaaaaaaa","".join(predicted_intents[i])+"".join(predicted_slots[i]))
                print("bbbbbbbbbbbbbbbbbbb","".join(target_intents[i])+"".join(target_slots[i]))

                #print(" ".join(predicted_slots[i]))

                target_semantics.append("".join(target_intents[i])+"".join(target_slots[i]))
                predicted_semantics.append("".join(predicted_intents[i])+"".join(predicted_slots[i]))

                print("")
            self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric_intents.append(
                    ids, predicted_intents, target_intents
                )
                self.wer_metric_slots.append(
                    ids, predicted_slots, target_slots
                )

                self.wer_metric_all.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_intents, target_intents
                )

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                with jsonlines.open(
                    hparams["output_folder"] + "/predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        dict["file"] = id_to_file[ids[i]]
                        writer.write(dict)

        return loss

    def log_outputs(self, predicted_semantics, target_semantics):
        """ TODO: log these to a file instead of stdout """
        for i in range(len(target_semantics)):
            print("".join(predicted_semantics[i]).replace("|", ","))
            print("".join(target_semantics[i]).replace("|", ","))
            print("")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()
        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:

            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric_intents = self.hparams.error_rate_computer()
            self.wer_metric_slots = self.hparams.error_rate_computer()
            self.wer_metric_all = self.hparams.error_rate_computer()
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric_intents.summarize("error_rate")
            stage_stats["SER_intent"] = self.wer_metric_intents.summarize("SER")
            stage_stats["SER_slots"] = self.wer_metric_slots.summarize("SER")
            stage_stats["SER_all"] = self.wer_metric_all.summarize("SER")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["SER_all"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["SER_all"])

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec2_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"SER_all": stage_stats["SER_all"]}, min_keys=["SER_all"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("intents")
    @sb.utils.data_pipeline.provides(
        "intent", "token_intents_list", "tokens_intents_bos", "tokens_intents_eos", "tokens_intents"
    )
    def text_pipeline_intent(intents):
        yield intents
        tokens_intents_list = tokenizer.encode_as_ids(intents)
        yield tokens_intents_list
        tokens_intents_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_intents_list))
        yield tokens_intents_bos
        tokens_intents_eos = torch.LongTensor(tokens_intents_list + [hparams["eos_index"]])
        yield tokens_intents_eos
        tokens_intents = torch.LongTensor(tokens_intents_list)
        yield tokens_intents
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline_intent)
    @sb.utils.data_pipeline.takes("slots")
    @sb.utils.data_pipeline.provides(
        "slots", "token_slots_list", "tokens_slots_bos", "tokens_slots_eos", "tokens_slots"
    )
    def text_pipeline_slots(slots):
        yield slots
        tokens_slots_list = tokenizer.encode_as_ids(slots)
        yield tokens_slots_list
        tokens_slots_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_slots_list))
        yield tokens_slots_bos
        tokens_slots_eos = torch.LongTensor(tokens_slots_list + [hparams["eos_index"]])
        yield tokens_slots_eos
        tokens_slots = torch.LongTensor(tokens_slots_list)
        yield tokens_slots
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline_slots)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_slots_bos","tokens_intents_bos", "tokens_slots_eos","tokens_intents_eos", "tokens_slots","tokens_intents","intents","slots"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing SLURP)
    from prepare_multihead_pred import prepare_SLURP  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_SLURP,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "slu_type": "direct",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    print("Creating id_to_file mapping...")
    id_to_file = {}
    df = pd.read_csv(hparams["csv_test"])
    for i in range(len(df)):
        id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])


