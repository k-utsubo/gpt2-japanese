import json
import os
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
from tensorflow.contrib.training import HParams
import model
from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/Users/admin/data/gpt2ja-medium')
parser.add_argument('--context', type=str, default="世界気象機関（ＷＭＯ）は１４日、２０１６年の世界の平均気温が１５年の記録を上回り、観測史上で最も高くなるとの見通しを正式に発表した。　１６年はハイチや米南部を襲ったハリケーン「マシュー」をはじめ、世界各地で異常高温や洪水、干ばつなどが相次いだ。ＷＭＯは「人間が引き起こした地球温暖化がこうした異常気象の背景にある」と分析。今月発効した地球温暖化対策の新枠組み「パリ協定」の下で、各国が温室効果ガスの削減を進めるよう求めている。　モロに合わせて発表した。　ＷＭＯによると、１６年１～９月の気温は産業革命前に比べて１・２度高く、パリ協定が掲げる「２度未満」の目標達成が危ぶまれる状況。北半球の高緯度地方が特に高温で、ロシアの北極圏は１９９０年までの３０年間の平均に比べて６～７度高かった。ＷＭＯのターラス事務局長は「１度未満の単位で気温を観測するのに慣れた身からすると異次元の事態だ」と警戒している。　世界の平均海面は２月までの１年余りで１・５センチ上昇。ここ２０年余りの平均的な年間上昇幅の約５倍の記録的なペースとたことが響いた。")
parser.add_argument('--gpu', type=str, default='-1')
args = parser.parse_args()

with open('ja-bpe.txt') as f:
    bpe = f.read().split('\n')

with open('emoji.json') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

if 'small' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12
    })
elif 'medium' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
elif 'large' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1280,
      "n_head": 20,
      "n_layer": 36
    })
else:
    raise ValueError('invalid model name.')

config = tf.ConfigProto()
if int(args.gpu) >= 0:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu
with tf.Session(config=config,graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    context_tokens = enc.encode(args.context)
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })
    output = out['h_flat'][-1]
    print(output.tolist())
