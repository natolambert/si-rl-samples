# si-rl-samples

Summary:
--------

There is a critical need to capture crisply the idea that feedback reduces uncertainty, and that when one is permitted to act in feedback, one can tolerate imperfect models far more readily than when one has to be open-loop.
Anyway, if you’re interested, there is a little project that we had identified last semester. Alex Devonport is also interested if you want to work with him. I would love to see this done.
Goal: consider a simple linear stabilization problem in closed loop for a system with a significant number of states that is controllable via scalar control and observable from a scalar output.
In the first corner: classical system-id plus eigenvalue-placement via observer and controller design, with a tuning loop on the outside that keeps learning continuously from the data.
In the second corner: Various flavors of model-free Deep-RL. Policy gradients, DQN, replay buffers, etc.
The question is: how dramatic is the difference in performance and sample complexity between these two.  Especially as the number of internal states increases.
We had thought that this would be very cool for students to see, and would be a nice paper regardless.

Dependencies:
-------------
1. https://github.com/python-control/python-control (need to add observer implementation for notion of sample efficiency)
2. A group of RL baselines - I am considering RLKit, but can we should include model-based algorithms? Not sure without different assumptions of state measurements.
  a) PETS for model-based RL (have a stable baseline)
  b) SAC for model-free RL, (have a stable baseline)

Plan of Attack:
---------------
1. Make a linear system that fits the bill (can try cartpole as one baseline https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) or create a system directly with matrices
2. Implement observer design (Try: https://github.com/python-control/python-control/blob/a4b4c43e51f0fc2cbf389336a90230a6a741c0dc/external/yottalab.py#L214)
3. figure out this: with a tuning loop on the outside that keeps learning continuously from the data
4. Run experiments.


Instructions:
-----------------
1. Create initial conda env
```
conda env create -f environment.yml
pip install control 
pip install slycot
conda activate samples
```
2. Install dependencies (anywhere on computer)
```
git clone https://github.com/vitchyr/rlkit
cd rlkit 
pip install -e .
```

misc experiment ideas:
----------------------
1. In real examples, RL works by repeated episodes and SI/Control works by identifying the states over time. What poles of the observer / controller result in a system that RL can solve but these cannot over short intervals.
2. Do we want to consider how fast the observer converges, or the speed at which SI works. It should be the latter, my bad.
