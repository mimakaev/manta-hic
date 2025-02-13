import glob
import os

import click
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from ushuffle import shuffle

from manta_hic.nn.manta import Manta2, fetch_tile_microzoi_activations
from manta_hic.nn.microzoi import MicroBorzoi
from manta_hic.ops.seq_ops import open_fasta_chromsizes

seqq = [
    "CTATCTCTTCCCCATTCTGAGTAATGGGAACTCACTTTATTTTGCACATAAATCCAAATACAAGAAGAAGAGGAATGTTAGGCTATGTCTGCAGTACAAATTACCTAGGGAA",
    "AAGAAAATTAAGATGACTTCTAAGGATGAAAAATAATCTTCCCAACAGAGTTCAAAAGGACTCAGAAATGGTCTCTATGGGGAAAGCTTTCATTAAACAGAAAAACATAAAA",
    "GAGGCGACAGTCCTGCCCTGAGAGGGTAATGGCCAATTTGACCTATTAGGTGTTTACTGCATCAAGTTTATTTGTTGCCCTGATTAGAAACCACTAAGGGGACATTGGAAAT",
    "GGGGGAGGACTGTCTAGGCCACAGGCATTCCTTCTAGATTCTGTCTTTTGAGGACTGGTCCCCACACAATGTTTCTAGGGTTGATAAACTCACCTCTTTTGACCTTCAGCCT",
    "TCCCTAGGATGACAGGATATTGAAGGAAATATATATTTGTAGTTGGTGGTAGTGAGGAAAAATATAAGATAAACTTCAGGTGCAATAAAAGGGAGTTGAAGTAGGAACATCC",
    "GAAAATTAATTTCCTGGCTCCAAGTGTTATTGGATTGTAGCTCAACAATCCACTTGGGACATGTTAGAAGCATATCATGGAAAATAGGTGAGCTGAGTGGATGGAAACACCA",
    "AATGATGAGTAATAGAACAATATCTGTATAGGCAAAGATCCTAGTAACCCCTGGATACCTTATTTGTCTGTGTCTATGGTGCAGAAGATTTTATGTTACCTGCTGTCCAGCT",
    "GCCTCAAATGAATCAACTGTTTAGAAAGCTTATTTACATTTTAATAGGAGGCCCAGGCATCTGTTCCCAGCACCTAGTGCCTCCTGCAGCATGCTTGGGGCTGTCCTGGAAG",
    "CTGCTTGTTGATTCCTGCAGGGGCTTGATTGTCTCATCAGATTATTAGAATTAACTGGGCAAAATCAGATTTCAAACTGGATTAAGACATCAAGAAAATGCTGCTGACCAAT",
    "TTCTCTTCTCCCAGATCAGAAAAAAATGCCTATAGATGCAAGATACTTCATAAATGTAAATAATTAACCCTACATATTTGTACATACATTACAATTTACAAAACACTTCTTT",
    "AAATGTTGTGTTTAGATCCATGTGAACTCCTTGTAAAGTAGGCAGAGTATATGTTATTGCACAAATAATCATCTACAGATGAGGAAACCGAGGCTTGATTGCCTAATGTCAC",
    "TAAAGAACAAAGTCAAAGCTAAGTATTCTGACCTCCAGAATACCACTGATATCTAACCCGTAGCCAAGAATACACAAGCATTTTGAAGCTGTGAATGTTTGTCATTACCCTT",
    "CACATTTTCTTTTTCTGTGGGAATGGAAGGTTAAGATGACATTGACTATGATGGTACAGCTCTTTAGAAGGCACTTTGGAAAGTACAGGTTCAGTCATGGTCACAGGCCTGC",
    "AAGAACTTTAATCTGGTTTGGGAATAGACATACAACCATTGAATCTTTCATAACTCTCAAAGGGGATATCTGAATATATGATGAAATGACAGCAGAGATAAAACTACTAGTG",
    "AAATTCCAACTAATAACATGTTATTATGAGCTAGAGAATAAGCAGAAGGTTTTGTCATGAAAGCGTCTCACAAAGATGTGCAGAATGACAGGACTTATGGACAAAGAAGGTA",
    "AGTGGGGTGGGAGATTCCAATGAAGATATAATTTGAGCAAACATATGGAGTGGGCTATATCTTTTAGCACCATAATAATGGGTATTAGAGGCTGAAATGAAACCTGTAGAGT",
    "AGATTTAAGAGAGGTAGCTCTGAAACCAGATTGTTGGAGTTCAAATCTCCATCACTCACTGAGCAAGCTATGTGACAATTTCTCTATCTGTCTCTGGGGATAAGAATAGTAC",
    "CAACCAACCTCACAGAATTTTGTGATGATTAAATGAGATGGTACATTTACAAATCCTTCATTAAAATAGTACCTATACTTAGTTGCCATTTTAAAACAAAACAAAACTGTGT",
    "TATTGGTTTTAAAAAAATGTCAAGAGCTAAGGAGGTGGTTAGATTTGGGAGTATAAATAGAGTGAACATATGATTTTTTATTCAAGTCTTTTAGCCTTTGAAGCGAAAGGAT",
    "CTGCTATTAATGCTTATGCCAGGACAGTAAGTATAGAGCAGTCAATGGCTTGCATAAACCAGGTGTGGTGGCCCTGCATACAGAAAACCTTACTCTAAATTTTCTGTTGTTG",
    "TTTATTCAGACTTCCCAGGTATGTCACAAACTTTCATGATCAGATAGCACTACAGAGGTCAGGATCTTGTAAGCATTTTCCATAATATAATCCAAAGTGTTGAGCTTTGGAT",
    "CAAGAATGTCCAAAAGATAAAGGATCTTAAAGATCACCTAATATTCCTTCTACATTTTCAGTTTTGGAAGGAATCTCATTGAATTCAGGTGCTGACACAATGTCTGTGGTTT",
    "CATTAGGCAACACTTATATAAAGTCCACCTAGAACTGGACAGGAGTAGCCTACACACTCATTGGAGACCATTTTACAGATTTCAAATTATTATGAAAAGTCCAGGGCTTTTT",
    "ATACAGATGAGGAAATGGTTCCAGACTGGCTATGGGCCATATTCAAGATCTCTTAATATCCAGTTTGGTTTGTTTTTAACACAACAGTGTTTCCCAAAATATGTTTTATAGA",
    "ACATGAGTCCTGCAGGATGCTCTGGCAGGGAGGGAGAAAGGGCAAGGAAGACCATGGCTGAATAAGCTTTGGGTATGATGCATATTATGTTATATTTTTGGAAATTCAGATT",
    "ATACACTAAGTACTCTGAGAAGTCCTTGAAAAACAACTTAATCTTGTTCAAAGGAGCATTTCCCATATTTGTTTTACCATGAAACCTTTGTACAATACAAATTAACGTTCCG",
    "CACAATTAGTGTTCTGATGTTACATGCTTTGTAAGGGACTACACTGTATAATTTAGTACGACATGAAGAATTTCCAAACCAGTAAGAAGATGTAAATGGATCATTAGAACAT",
    "TTTTGGCTTTCAAATAGATTTACATGTAGTCATAAAATCAACATCTCTCCAGTCAAATTCATTCTTATGGTTGTTTTATGTACCATATTTAATCAAGCTTATTTTTGTATTT",
    "TTTGCCCAATTAATCATTTTTTCATTATTGAATAGTTTGCTTTAGCTTCTGTCAAAGTAGCAATATTAATAGGATAAAGTAATTCTGGCTCAATTAAATATTGTATAACCAT",
    "GAAAAAAACGGGGAGAAAATACTTTAATATTTATTCAAACCTATAGAGTGATGAGGACTTTTGAATTTTCAAAACAATGGGAGAATTTAGAAAAAGAAAAGGATTAACTGTG",
    "TAAAAATGTTGTATTTTCCCATGTCAAAAAAAGCCAATTTAGGAAATAGTAAAAAGGATTTATAGTTAAATGTGACAGCAAGTGATACCTATGTTAAATAAGGAAATGACTC",
    "ACATAAATAGAAAAAAGGCATACACACTAATAGCAACTACAAAAGTCTCTAAGAGAACAATGGAAAAAGAACCGACGACATGAACAGAGGAAGTAGAAATAGAATGAATAAA",
    "GGTGCCACTTGACTTCCCTACATCCAGCTTCTCTACAGGCCACACTTTCCCCTGTATGCACTGATCCTGCTAAGTCACTTTTCATACACCCCTTTCCACTCTCCATTACAGG",
    "CTGAGTAATGGTCCTGAGTGATCTGCAACCTAGGGCTACAGCACCCACAGGAGCCATCATTAGTTATTTTATAATAGTTTTTGTTCTCTGAAAGGGAACTTCTCCTTATGTC",
    "TACAAGTCTATCGTTCAACTTAGAACCATTCAATTTGCAACACATATTTAAGAGTATATTATATACCAAAAAAGGGACTTTTTGTAGACATAGTGACCTTGGAGTAAACTTC",
    "CTTGTAAGTGTAGCCTTCCAGGTTCCCAGGACTTCAGTGCATGAGCTCATTCAAGAAATAGACCAGGGCCAGCCGCGGTGGCTTACGCCTGTAATCCCAGCACTTTGGGAGA",
    "CTGAGGTGGGCAGATCACAAGGTCAAGAGTTTGAAACCAGCCTGGCCAATATGGTGAAACCTCATCTCTACTAAATATACAAAAATTACCCGGGCATGATGGCACGCACCTG",
    "TAGTCCCTCCTACACAAGAGGCTGAGGCAGAAGAATTGCTTGAACCCAGGAGGTGGAGGTTGCAGTGAGCCACGATCGTGCCACTGCACTCCAGCCTGGGTGATAGAGCGAG",
    "ACTCCATCTCACAAAAAAAAAAAAAAAAAAAAGAGAGAGAGAGAGACACCACATAAAACTTTTATCTCGAACTGTCACCTCTGCTCTCTGTTGGTAGTACTTGATTAGCTTG",
    "CCTTCACATTTCTGGTACTGTCTTATGAAATTTTCACTGTTCATCTTCTGAGAAGACATAAGCCCCTGGTAACATGAGACAGTCTGTCATCCTCTATCTGAGGCACACATGT",
    "CAATCAATCTTCTCCTATGTTTCCCAGAGAGGGAATTTAATTACATTTCCTGTCTATCTCTGTTATCAATTGCTGCAAAATAAACCACTTCCACATTTTGTGTCTTAAAACA",
    "ACAATTTATTATTATC",
]
seqq = "".join(seqq).encode()


def make_quiescent_seq(seq_len):
    stmp = shuffle(seqq, 4).decode()
    st = len(seqq) // 2 - seq_len // 2
    ed = st + seq_len
    if ed >= len(stmp):
        raise ValueError(f"Cannot make sequences longer than {len(stmp)}")
    return stmp[st:ed]


@click.command()
@click.option("--device", default="cuda:0")
@click.option("--filename", required=True)
@click.option("--out_filename", required=True)
@click.option("--microzoi_file", required=True)
@click.option("--fasta_file", required=True)
@click.option("--model-wildcard", required=True)
def manta_mutate_file(device, filename, out_filename, microzoi_file, fasta_file, model_wildcard):

    if os.path.exists(out_filename):
        return

    open(out_filename, "w").close()

    # Open FASTA
    fa = open_fasta_chromsizes(fasta_file)[0]
    microzoi = MicroBorzoi(return_type="mha")
    st_dict = torch.load(microzoi_file, map_location=device, weights_only=True)
    microzoi.load_state_dict(st_dict, strict=False)  # we are not loading the final keys
    microzoi = microzoi.to(device)
    microzoi.eval()

    models = {}

    for i in glob.glob(model_wildcard):
        if "downsamp" in i:
            continue
        st_dict = torch.load(i, weights_only=True, map_location=device)

        name_filename = os.path.join(os.path.split(i)[0], "names.txt")
        names = [i.strip() for i in open(name_filename).readlines()]
        names = [i for i in names if i]

        output_channels = st_dict["final_conv.weight"].shape[0]
        manta = Manta2(output_channels=output_channels, tower_height=2)
        manta.load_state_dict(st_dict)
        manta = manta.to(device)
        _ = manta.eval()
        name = i.split("/")[-2]
        models[name] = (manta, names)

    def quantify_difference(x, name_list, model_name, record_list, offset=0.01):
        logratio = ((x[::2] + offset) / (x[1::2] + offset)).log10()

        mut_pos = 512
        for off in 0, 3, 10, 30, 100, 200:
            changes = {}
            if off == 0:
                changes["all"] = logratio
            else:
                slice_at = slice(mut_pos - off, mut_pos + off + 1)
                changes["betw"] = logratio[:, :, mut_pos - off : mut_pos, mut_pos + 1 : mut_pos + 1 + off]
                changes["up"] = logratio[:, :, slice_at, mut_pos + off + 1 :]
                changes["down"] = logratio[:, :, slice_at, : mut_pos - off]
                changes["at"] = logratio[:, :, slice_at, slice_at]

            means = {i: j.mean(axis=(2, 3)).cpu().numpy() for i, j in changes.items()}
            abses = {i: j.abs().mean(axis=(2, 3)).cpu().numpy() for i, j in changes.items()}
            means_sq = {i: (j**2).mean(axis=(2, 3)).cpu().numpy() for i, j in changes.items()}

            sample = list(means.values())[0]

            for rep in range(sample.shape[0]):
                for name_ind in range(sample.shape[1]):
                    for ar, avg_name in [(means, "mean"), (abses, "mean_abs"), (means_sq, "mean_sq")]:
                        for metric in means.keys():
                            record = {"chrom": chrom, "start": start, "end": end}
                            record["rep"] = rep
                            record["name"] = name_list[name_ind]
                            record["metric"] = metric
                            record["avg"] = avg_name
                            record["offset"] = off
                            record["value"] = ar[metric][rep, name_ind]
                            record["model_name"] = model_name
                            record_list.append(record)

    in_df = pl.read_parquet(filename)

    hic_res = 2048
    pad = 128
    manta_bp = hic_res * (1024 + 2 * pad)

    all_dfs = []
    for row in in_df.iter_rows(named=True):

        chrom = row["chrom"]
        start = row["start"] - 50
        end = row["end"] + 50

        mutates = [[("replace", start, make_quiescent_seq(end - start))] for _ in range(4)]

        mid = (start + end) // 2
        reg_start = mid - manta_bp // 2 - hic_res // 2  # this puts us in the center of 512th bin
        reg_end = reg_start + manta_bp

        acts = []

        with torch.no_grad(), torch.autocast(device):
            for i in range(4):
                crop_mha = np.random.randint(640, 786)
                offset_bins = np.random.randint(0, crop_mha)
                shift_bp = np.random.randint(-128, 128)
                a1 = fetch_tile_microzoi_activations(
                    microzoi,
                    fa,
                    chrom,
                    reg_start,
                    reg_end,
                    start_offset_bins=offset_bins,
                    shift_bp=shift_bp,
                    crop_mha_bins=crop_mha,
                    batch_size=4,
                )
                a2 = fetch_tile_microzoi_activations(
                    microzoi,
                    fa,
                    chrom,
                    reg_start,
                    reg_end,
                    start_offset_bins=offset_bins,
                    shift_bp=shift_bp,
                    crop_mha_bins=crop_mha,
                    mutate=mutates[i],
                    batch_size=4,
                )
                acts.append(a1)
                acts.append(a2)
            records = []
            for model_name, (model, names) in models.items():
                res = torch.cat([model(torch.stack(acts[:4])), model(torch.stack(acts[4:]))])
                quantify_difference(res, names, model_name, records)
        all_dfs.append(pl.DataFrame(records))
        print(chrom, start, end)
    megadf = pl.concat(all_dfs).with_columns(cs.integer().cast(pl.Int32), cs.float().cast(pl.Float32))
    megadf.write_parquet(out_filename, compression_level=16)
