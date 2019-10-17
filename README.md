# stabalizable-learning-control

Summary:
--------

There is a critical need to capture crisply the idea that feedback reduces uncertainty, and that when one is permitted to act in feedback, one can tolerate imperfect models far more readily than when one has to be open-loop.
Anyway, if you’re interested, there is a little project that we had identified last semester. Alex Devonport is also interested if you want to work with him. I would love to see this done.
Goal: consider a simple linear stabilization problem in closed loop for a system with a significant number of states that is controllable via scalar control and observable from a scalar output.
In the first corner: classical system-id plus eigenvalue-placement via observer and controller design, with a tuning loop on the outside that keeps learning continuously from the data.
In the second corner: Various flavors of model-free Deep-RL. Policy gradients, DQN, replay buffers, etc.
The question is: how dramatic is the difference in performance and sample complexity between these two.  Especially as the number of internal states increases.
We had thought that this would be very cool for students to see, and would be a nice paper regardless.

Plan of Attack:
---------------


Additional Notes:
-----------------