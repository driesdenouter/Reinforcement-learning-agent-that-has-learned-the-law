# Using reinforcement learning to train an autonomous vehicle to avoid obstacles and obey the speed limit

This is a project I have used for my Master Thesis in Tax Law. The goal was to find out if it is possible to learn a reinforcement agent to obey the law, in this case not violating the speed limit. I have set the speed limit to 30. The majority of the code has come from the three medium articles from Matt Harvey that are listed below. 

In the original code the agent was not able to change its speed. I have adjusted the code so that it was able to do that. Eventually regarding the max traveled distance, the new agent preformed the same way as the original agent, with an average max distance without crashing off about 540 frames. And it drives at a speed of 29/30, so right at the allowed limit. 

Medium articles Matt Harvey:

*Part 1:* https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6

*Part 2:* https://medium.com/@harvitronix/reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-part-2-93e614fcd238#.vbakopk4o

*Part 3 (for this version of the code):*
https://medium.com/@harvitronix/reinforcement-learning-in-python-to-teach-an-rc-car-to-avoid-obstacles-part-3-a1d063ac962f

*Github link original code
https://github.com/harvitronix/reinforcement-learning-car

## Installing

These instructions are for a fresh Ubuntu 16.04 box. Most of the same should apply to OS X. 

### Basics

Recent Ubuntu releases come with python3 installed. I use pip3.7 for installing dependencies so install that with `sudo apt install python3-pip`. Install git if you don't already have it with `sudo apt install git`.

Then clone this repo at my Github account: driesdenouter. Project name: Reinforcement learning agent that has learned the law

### Python dependencies

`pip3.7 install numpy keras h5py`

That should install a slew of other libraries you need as well.

### Install Pygame

Install Pygame's dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

Then install Pygame itself:

`pip3.7 install hg+http://bitbucket.org/pygame/pygame`

### Install Pymunk

This is the physics engine used by the simulation. It just went through a pretty significant rewrite (v5) so you need to grab the older v4 version. v4 is written for Python 2 so there are a couple extra steps.

Go back to your home or downloads and get Pymunk 4:

`wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz`

Unpack it:

`tar zxvf pymunk-4.0.0.tar.gz`

Update from Python 2 to 3.7:

`cd pymunk-pymukn-4.0.0/pymunk`

`2to3.7 -w *.py`

Install it:

`cd ..`
`python3.7 setup.py install`

### Run learning.py to start train process. 

![](https://github.com/driesdenouter/Reinforcement-learning-agent-that-has-learned-the-law/master/RL-agent (2).gif)
