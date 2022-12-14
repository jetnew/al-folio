---
layout: post
title: "Getting Started with Reinforcement Learning"
description: A curation of resources for reinforcement learning.
date: 2021-07-07
comments: research
giscus_comments: true
---

## What is Reinforcement Learning (RL)?

You may have heard about the successes of artificial intelligence (AI) over humans in games such as Go ([Go master quits because AI 'cannot be defeated'](https://www.bbc.com/news/technology-50573071)), StarCraft ([DeepMind's StarCraft 2 AI is now better than 99.8 percent of all human players](https://www.theverge.com/2019/10/30/20939147/deepmind-google-alphastar-starcraft-2-research-grandmaster-level)) and DoTA 2 ([OpenAI's Dota 2 AI steamrolls world champion e-sports team with back-to-back victories](https://www.theverge.com/2019/4/13/18309459/openai-five-dota-2-finals-ai-bot-competition-og-e-sports-the-international-champion)).

The core technology behind these AI successes is reinforcement learning. Reinforcement learning is a [machine learning](https://en.wikipedia.org/wiki/Machine_learning) paradigm (alongside supervised and unsupervised learning) that learns an optimal sequence of actions to take in the environment. In contrast, [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) learns a single output prediction based on an existing training dataset of labels, and [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) learns the distribution of data without any given labels.

As you may have expected, reinforcement learning is closely related to (and is in fact inspired by) the same concept behind [Pavlov's Dog](https://en.wikipedia.org/wiki/Classical_conditioning), operant conditioning. [Operant conditioning](https://en.wikipedia.org/wiki/Operant_conditioning) is a learning process that associates and modifies behaviours with reinforcement or punishment. This is well described by psychologist [Edward Thorndike](https://en.wikipedia.org/wiki/Edward_Thorndike)'s [Law of Effect](https://en.wikipedia.org/wiki/Law_of_effect), which states:

> "Responses that produce a satisfying effect in a particular situation become more likely to occur again in that situation, and responses that produce a discomforting effect become less likely to occur again in that situation."

The process of reinforcement learning takes place in 4 stages:
1. The environment's state is first observed by the agent.
2. The agent takes an action based on the observation.
3. The agent's action influences the environment's state.
4. The environment then returns a new state to the agent, along with a reward indicating the goodness of the action taken.

Reinforcement learning is currently used by a few applications in industry, namely [recommending personalised advertisements](https://venturebeat.com/2021/02/23/how-reinforcement-learning-chooses-the-ads-you-see/), improving search engines, and more recently, [improving pricing algorithms](https://engineering.grab.com/understanding-supply-demand-ride-hailing-data) of ride-hailing applications based on live demand and supply. While RL currently does not have as many applications as domains like [computer vision](https://en.wikipedia.org/wiki/Computer_vision) and [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing), RL is a promising and upcoming method with high potential in fields like [robotics](https://bair.berkeley.edu/blog/2020/04/27/ingredients/) (think automating thousands of manual processes in factories), self-driving vehicles (think reducing accident rates while significantly improving logistics transportation efficiency) and, of course, video games (who doesn't enjoy seeing games being solved/mastered?)

Reinforcement learning can seem quite intimidating to beginners due to the vastness of the field. In this blog article, I will share the resources and some advice on how to use them effectively to get started with learning reinforcement learning. These recommendations are heavily opinionated based on my own learning journey in RL.

***

## Introductory Reinforcement Learning

Reinforcement learning is a sub-field of machine learning, and so learning RL will require learning machine learning. You should also learn RL in consideration of mathematical theory and practical coding. The following 3 resources are thus recommended based on those considerations.

### 1. [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro) by Google

Reinforcement learning requires many concepts from machine learning (ML). This crash course (that contains mainly explanatory videos) provides an introductory understanding to the main ML concepts that you will need for RL. Some concepts covered: Training vs Validation vs Testing, Classification vs Regression, Introductory Neural Networks.

### 2. [Reinforcement Learning: A Tutorial](http://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf) by Harmon & Harmon

This tutorial presents the theory of RL at an introductory level. It formalises the RL paradigm with math, and is a great resource so that concepts are rigorous and not too hand-wavy. The math and notation may make it look intimidating, but if you understand the underlying motivation and concepts, understanding the notation will come naturally. Follow through thoroughly and you will be rewarded well. Topics covered: Dynamic programming, Value iteration, Q-learning, Temporal difference learning, Discounting rewards. Nonetheless, even if you're unable to follow through, you should still face no problem proceeding with the next resource on Medium.

### 3. [Deep RL Course on Medium](https://simoninithomas.github.io/deep-rl-course/) by Thomas Simonini

This Medium series on deep RL (deep because it uses deep neural networks) is one of the best courses for learning introductory deep RL with a good balance of practical coding and RL theory. Topics covered: Deep Q-learning, Policy Gradients, Actor Critics, Unity-ML Agents. The resources up to actor critics are excellent; Learning Unity-ML Agents is optional, learn it if you are interested in the Unity-ML Agents environments.

***

## Tools for Reinforcement Learning

### 1. RL Gym Environments

RL Gym environments are the simulation software that is used to train RL agents in. They're called "gyms" because of [OpenAI gym](https://gym.openai.com/)'s standardisation of environment APIs. There are too many RL environments, but here are some lists to refer to: [awesome-rl-envs](https://github.com/clvrai/awesome-rl-envs), [env-zoo](https://github.com/tshrjn/env-zoo) and [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl). The RL environments are diverse and many are really cool, such as IKEA furniture assembly, Retro games and autonomous vehicle simulators.

### 2. RL Frameworks

RL frameworks are frameworks made specifically for building RL agents and/or training them. While most RL research work use general deep learning frameworks, e.g. [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [JAX](https://jax.readthedocs.io/en/latest/index.html), [FLAX](https://github.com/google/flax), [Pytorch Lightning](https://www.pytorchlightning.ai/), [TensorFlow Probability](https://www.tensorflow.org/probability), there are also RL frameworks available. There is currently no established "best" RL frameworks yet due to the diversity of RL concepts, but I found [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/) (recently version 3) very useful for basic usage of RL. [TensorFlow Agents](https://www.tensorflow.org/agents) is a library used by some companies in industry. [minimalRL](https://github.com/seungeunrho/minimalRL) and [CleanRL](https://docs.cleanrl.dev/) are repositories of minimal code implementations of RL algorithms in PyTorch that is useful for learning.

***

## Intermediate Reinforcement Learning

Reinforcement learning is a vast field that contains many different concepts (many inspired by cognitive neuroscience). Behaviours are influenced and controlled by many mechanisms and so many methods are available in reinforcement learning. This section will provide some tips on exploring the many domains in RL, e.g. Exploration, Memory, Model-based. It is important to note the difference between "RL" and "deep RL", the former being about traditional RL while the latter about applying neural networks to RL.

### 1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Richard S. Sutton and Andrew G. Barto

Sutton and Barto is a good **textbook** on RL that is commonly coined the RL "bible" for its prominence. The textbook is a comprehensive resource and reference for RL, covering tabular methods such as multi-armed bandits, dynamic programming, Monte Carlo methods, TD-learning and bootstrapping, to approximate methods for prediction and control and policy gradients.

### 2. [Introduction to Reinforcement Learning](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) by David Silver, DeepMind

Intro to RL by David Silver, DeepMind is a good **lecture** series on reinforcement learning. It formulates RL with concepts about Markov decision processes, planning by dynamic programming, prediction vs control, value function approximation, among others. David Silver is a RL pioneer who co-authored [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), one of the first successful attempts to apply deep neural networks to reinforcement learning.

### 3. [Reinforcement Learning Specialization on Coursera](https://www.coursera.org/specializations/reinforcement-learning) by University of Alberta

The RL Specialization on Coursera is a good **course** on RL, introducing concepts such as temporal difference learning, Monte Carlo, Sarsa, Q-learning, policy gradients, the Dyna architecture, and more. The course is taught by Martha White and Adam White, who are students of Richard Sutton, the author of the Reinforcement Learning: An Introduction textbook ([who personally recommended the course](https://www.reddit.com/r/MachineLearning/comments/h940xb/what_is_the_best_way_to_learn_about_reinforcement/)).

### 4. [Spinning Up: Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) by OpenAI

Spinning Up by OpenAI is a resource that introduces deep RL concepts to people interested in picking up RL. Their Key Papers in Deep RL section provides a good starting point to the overview of RL topics, introducing milestone research papers in each topic. Examples: Model-Free RL, Exploration, Transfer and Multitask RL, Hierarchy, Memory, etc.

### 5. [CS285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) by Sergey Levine, UC Berkeley

CS285 is a good **lecture** series on deep reinforcement learning, with concepts covering policy gradients, actor-critics, model-based RL, exploration, offline RL, inverse RL and meta-learning in RL, among others. Offline RL is used to train an RL agent offline (i.e. collect a batch of interactions then training the agent on it). Inverse RL involves using interaction data from a human expert and is used to infer the RL policy underlying the human decision making process. Sergey Levine is a prominent RL researcher and his course provides many good insights into deep RL.

### 6. [Algorithms for Decision Making](https://algorithmsbook.com) by Mykel J. Kochenderfer, Tim A. Wheeler and Kyle H. Wray

Algorithms for Decision Making is a good and updated (2022) **textbook** that serves as a modern perspective of the problems in the field of reinforcement learning. In particular, it delivers a detailed walkthrough of belief state planning and multi-agent systems compared to Sutton & Barto, and contains beautiful illustrations for explanations, complete with code implementations in Julia.

***

## Advanced Reinforcement Learning

Beyond the standard learning resources are the topics that are active and unsolved open research questions. At this point, the research community is your best friend. The following resources and advice are strategies I found useful in my own learning journey.

### 1. Survey Research Papers

While there are many research papers that are impossible to read them all, there are survey papers that provide an overview of the current research landscape and topics. These survey papers provide an outline of the field and importantly introduce terminology that you can use to further search for what you are interested in. For example, [Reinforcement Learning: A Survey](https://arxiv.org/pdf/cs/9605103.pdf) (Kaelbling et al. 1996) surveys the field of RL from a computer science perspective. For a more updated survey on deep RL, refer to [A Brief Survey of Deep Reinforcement Learning](https://discovery.ucl.ac.uk/id/eprint/10083557/1/1708.05866v2.pdf) (Arulkumaran et al. 2017). I am personally interested in model-based reinforcement learning, so the following surveys served me well: [Model-based Reinforcement Learning: A Survey](https://arxiv.org/abs/2006.16712) (Moerland et al. 2020), [Deep Model-Based Reinforcement Learning for High-Dimensional Problems, a Survey](https://arxiv.org/abs/2008.05598) (Plaat et al. 2020). If you are interested in other domains in RL such as exploration or memory, you can search for terms terms like "reinforcement learning exploration survey" or "reinforcement learning memory survey" on Google Search or [Google Scholar](https://scholar.google.com/).

### 2. Research Blogs / Articles

Blog posts are my personal favourite for developing an initial understanding of the research landscape or particular research papers. One of the most popular blogs is [Lilian Weng's blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html), that covers many topics including RL, natural language processing, computer vision and representation learning. [Berkeley AI Research (BAIR)'s blog](https://bair.berkeley.edu/blog/), [Google AI Blog](https://ai.googleblog.com/) and [Facebook AI Research (FAIR)'s blog](https://ai.facebook.com/blog/?page=1) often blog about their RL papers. [Distill](https://distill.pub) was a peer-reviewed journal (not a blog) that focuses on interactive articles about machine learning research topics, allowing for playable exploration on their articles.

### 3. YouTube Channels

YouTube channels also provide good videos that explain papers or provide updates on the state of research. [Yannic Kilcher](https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew) is a well-known YouTuber for his videos that explain recent hotly discussed papers. [Henry AI Labs](https://www.youtube.com/channel/UCHB9VepY6kYvZjj0Bgxnpbw/videos) provide weekly updates about AI in general. [Machine Learning Street Talk](https://www.youtube.com/channel/UCMLtBahI5DMrt0NPvDSoIRQ) conducts interviews with researchers to discuss about their research.

### 4. Online Communities

Various online communities are active for research discussions as well, such as communities on Twitter or Reddit. Online communities are important because they foster discussion, debate and ideation. Twitter (or "Academic Twitter") is a highly active community because many researchers tweet about their research. A list of researchers to follow on Twitter: [here](https://www.reddit.com/r/MachineLearning/comments/b6dwrd/d_best_current_twitter_accounts_to_follow_for/), but do actively adjust your Twitter feed according to your research interests. [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) is an active community that posts interesting research papers or discussion topics about ML research in general with thought-provoking comments. Or you could start your own a reading group!

***

## Conclusion

Reinforcement learning, as you now know, is such a vast and diverse field by nature that is still a highly active research area with many open problems. While RL is currently mainly used for ads, search and recommendation, RL holds great potential by unlocking automation of robotics, driverless vehicles and many other applications. With this article, I hope that it helps more people get into reinforcement learning and unlock the many things that can be done with RL. That said, again, this is an opinionated article based on my learning journey, and likely will change as I progress in my learning. If you have recommendations for resources or tips to include in this article, feel free to contact me at [notesjet@gmail.com](mailto:notesjet@gmail.com). Cheers!