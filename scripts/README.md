
# Notes
---

## Resources:
* Abstract class for multi-agent environments: [[1]](https://github.com/HumanCompatibleAI/adversarial-policies/blob/baa359420641b721aa8132d289c3318edc45ec74/src/aprl/envs/multi_agent.py)
* Stable-baselines PPO self-play:
    * Symmetric only (pytorch): [[2]](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py)
    * Symmetric only (tensorflow): [[3]](https://github.com/HumanCompatibleAI/adversarial-policies/blob/99700aab22f99f8353dc74b0ddaf8e5861ff34a5/src/aprl/agents/ppo_self_play.py)
    * Vanilla PG self-play (no stable baselines): [[4]](https://github.com/mtrencseni/pytorch-playground/blob/master/11-gym-self-play/OpenAI%20Gym%20classic%20control.ipynb)
* MARL with RLlib, can handle asymmetric agents and has tensorflow & torch support: [[5]](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_cartpole.py)
    * Learning via self-play: [[6]](https://github.com/ray-project/ray/blob/master/rllib/examples/self_play_with_open_spiel.py)
    * Independent learning: [[7]](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py)


##### Starting point:
Essentially, for MARL with `PyTorch` support there are only the options of `RLlib` and `StableBaselines`. Other frameworks do not support (easy) vectorization and continuous type/action spaces. See the [PettingZoo paper](https://proceedings.neurips.cc/paper/2021/file/7ed2d3454c5eea71148b11d0c25104ff-Paper.pdf), Section 8 for a few less known frameworks.

We have "GPU support >> MARL": adding MARL afterward is easy, adding GPU support not. In the online research community, there is a clear preference for `StableBaselines` compared to `RLlib`. Notes on `RLlib`:
* is not easy to use, has a lot of overhead (in particular when custom implementations, such as our vectorized auction games or evolutionary strategies are needed),
* only shines for large scale (multi machine) deployment,
* [multi-agent feature](https://docs.ray.io/en/master/_modules/ray/rllib/env/multi_agent_env.html) does not support keeping all types/actions as a single tensor and then indexing (as we need).

**Conclusion:** Both frameworks have a limited capability for MARL (mostly symmetric self-play, centralized learning). Thus, we stick with SB and add a "self-play layer" similar or even adapted from our in-house `bnelearn` framework.

---
## Todos & Questions
* Perhaps it makes more sense to **not** use the `vectorized` feature of StableBaselines and just consider the batches as a single high-dim. action? This might reduce complexity, because SB loops over `num_envs` at multiple occasions - which we don't want. Essentially, this would increase the action space (by the batch dimension) and we would have to compress the rewards (over a batch) into a single reward.
    * Is the error backpropagated correctly then? Can we call the networks with different batch sizes? 
    * Does that mean we loss feedback? $\rightarrow$ I would argue no, because that's what is done in `bnelearn` with the intermediate stages still as additional information.
    * How much hacking would be needed? (E.g. some copying to CPU would have to be changed.)
