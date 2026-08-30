# ActorCritic架构

# 一、介绍

> 我们学习了用神经网络来拟合V(s)、Q(s,a)  
> 也学习了用神经网络来拟合$\pi(a|s)$  
> 将两者结合起来，就得到了**ActorCritic架构**

- 对应这部分内容

    ![RL_4](pngs/RL_4.png)

# 二、Actor

- 回忆[PolicyGradient](强化学习/PolicyGradient.md)中的**高方差问题**，两个解决思路：
    1. 利用`因果关系`，计算t时刻的$r(\tau)$时，做一些简化
        - 正好对应我们的Q(s,a)
    2. 引入`baseline`
        - 正好对应我们的V(s)

- 于是**Actor**部分的优化目标就可以改写为：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= E_{\tau \sim \pi_{\theta}(\tau)} \left[ \sum\limits_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) r(\tau) \right] \\
&= E_{\tau \sim \pi_{\theta}(\tau)} \left[ \sum\limits_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) [Q_\pi(s_t,a_t) - V_\pi(s_t)] \right] \\
&= E_{\tau \sim \pi_{\theta}(\tau)} \left[ \sum\limits_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A_\pi(s_t,a_t) \right] \\
\end{aligned}
$$

# 三、Critic

> 接下来的问题，我们如何预测$A_\pi(s_t,a_t)$呢？

$$
\begin{aligned}
A_\pi(s_t,a_t) &= Q_\pi(s_t,a_t) &- V_\pi(s_t) \\
&= r(s_t,a_t) + V_\pi(s_{t+1}) &- V_\pi(s_t)
\end{aligned}
$$

- 用神经网络来拟合V(s)，即可

# 四、回合制 --> 持续制

