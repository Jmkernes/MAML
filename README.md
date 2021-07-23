# MAML, MAML++

Ah so this is partially a lie. We also implement MAML++ as well.

The two papers are Model Agnostic Meta Learning (MAML) and How to train your MAML (MAML++).

Links to the two papers can be found at
1) https://arxiv.org/abs/1703.03400
2) https://arxiv.org/abs/1810.09502

Ok, so why make the repository? First, this isn't a full reproduction. We demonstrate that it works by using the toy sinusoid dataset. Second, if you hack pytorch and directly modify model attributes, you can write out a MAML loop in much much less code than other repos I've seen. No need to rewrite pytorch from scratch with Meta-layers that allow you to pass in any model weights to some model architecture skeleton.

It even works on BERT models. Try it!
