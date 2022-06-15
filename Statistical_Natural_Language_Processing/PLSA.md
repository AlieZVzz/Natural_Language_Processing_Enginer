> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.cnblogs.com](https://www.cnblogs.com/Determined22/p/7237111.html)

LSA（Latent semantic analysis，隐性语义分析）、pLSA（Probabilistic latent semantic analysis，概率隐性语义分析）和 LDA（Latent Dirichlet allocation，隐狄利克雷分配）这三种模型都可以归类到话题模型（Topic model，或称为主题模型）中。相对于比较简单的向量空间模型，主题模型通过引入主题这个概念，更进一步地对文本进行语义层面上的理解。

**LSA 模型**就是对词 - 文档共现矩阵进行 SVD，从而得到词和文档映射到抽象出的 topic 上的向量表示，[之前的一篇博客](http://www.cnblogs.com/Determined22/p/5780305.html)稍微提到过一点。LSA 通过将词映射到 topic 上得到 distributional representation（词的分布表示），进而缓解文档检索、文档相似度计算等任务中所面临的**同义词（多词一义）问题**：比如我搜索 “Java 讲义”，如果系统只是用字符匹配来检索的话，是不会返回一篇出现了“Java 课件” 但通篇没出现 “讲义” 这个词的文档的。所以说，单纯地从词 - 文档共现矩阵取出词向量表示和文档向量表示的向量空间模型，尽管利用了大规模文档集的统计信息，仍然是无法直接从 “语义” 这个层面上去理解文本的。但是 LSA 这种将词映射到 topic 上的向量表示，很难去应对**一词多义问题**：比如 “Java” 这个词既可能指编程语言，也可能指爪哇岛，即使这两种含义的 “Java” 都在文档集里出现过，得到的 LSA 模型也无法很好地区分。

关于 LSA ，这里稍微扯远一点，有一个挺有意思的演讲视频：[你的用词透露了你未来的精神状态](http://open.163.com/movie/2016/7/1/8/MBQEDI5K0_MBQEDMK18.html)（对应的 paper 在[这里](https://neuro.org.ar/sites/neuro.org.ar/files/Automated%20analysis%20of%20free%20speech%20predicts%20psychosis%20onset%20in.pdf)），用 LSA 来预测人的未来精神状态，相信看完这个视频之后一定会体会到科学的力量。

**pLSA 模型**是有向图模型，将主题作为隐变量，构建了一个简单的贝叶斯网，采用 EM 算法估计模型参数。相比于 LSA 略显 “随意” 的 SVD，pLSA 的统计基础更为牢固。

这篇博客就是整理一下 pLSA 模型的推导过程。

**pLSA：频率学派**

相比于 LDA 模型里涉及先验分布，pLSA 模型相对简单：观测变量为文档 $d_m\in\mathbb D$（文档集共 M 篇文档）、词 $w_n\in\mathbb W$（设词汇表共有 V 个互不相同的词），隐变量为主题 $z_k\in\mathbb Z$（共 K 个主题）。在给定文档集后，我们可以得到一个词 - 文档共现矩阵，每个元素 $n(d_m,w_n)$ 表示的是词 $w_n$ 在文档 $d_m$ 中的词频。也就是说，pLSA 模型也是基于词 - 文档共现矩阵的，不考虑词序。

pLSA 模型通过以下过程来生成文档（记号里全部省去了对参数的依赖）：

(1) 以概率 $P(d_m)$ 选择一篇文档 $d_m$

(2) 以概率 $P(z_k|d_m)$ 得到一个主题 $z_k$

(3) 以概率 $P(w_n|z_k)$ 生成一个词 $w_n$

概率图模型如下所示（取自 [2]）：

![](https://images2017.cnblogs.com/blog/1008922/201707/1008922-20170725234727044-657726408.png)

图里面的浅色节点代表不可观测的隐变量，方框是指变量重复（[plate notation](https://en.wikipedia.org/wiki/Plate_notation)），内部方框表示的是文档 $d_m$ 的长度是 N，外部方框表示的是文档集共 M 篇文档。**pLSA 模型的参数** $\theta$ 显而易见就是：$K\times M$ 个 $P(z_k|d_m)$ 、$V\times K$ 个 $P(w_n|z_k)$ 。$P(z_k|d_m)$ 表征的是给定文档在各个主题下的分布情况，文档在全部主题上服从多项式分布（共 M 个）；$P(w_n|z_k)$ 则表征给定主题的词语分布情况，主题在全部词语上服从多项式分布（共 K 个）。

**联合分布**

拿到贝叶斯网当然先要看看联合分布咯。这个贝叶斯网表达的是如下的联合分布：

$$P(d_m,z_k,w_n)=P(d_m)P(z_k|d_m)P(w_n|z_k)$$

$$P(d_m,w_n)=P(d_m)P(w_n|d_m)$$

假设有一篇文档为 $\vec{w}=(w_1,w_2,...,w_N)$ ，生成它的概率就是

$$P(\vec{w}|d_m)=\prod_{n=1}^N P(w_n|d_m)$$

我们看一下 $P(w_n|d_m)$ 的表达式。如果不考虑随机变量之间的条件独立性的话，有

$$P(w_n|d_m)=\sum_k P(z_k|d_m)P(w_n|z_k,d_m)$$

但是观察图模型中的 d 、z 、w 可以知道，它们三个是有向图模型里非常典型的 **head-to-tail** 的情况：当 z 已知时，d 和 w 条件独立，也就是

$$P(w_n|z_k,d_m)=P(w_n|z_k)$$

进而有

$$P(w_n|d_m)=\sum_k P(z_k|d_m)P(w_n|z_k)$$

所以最终的联合分布表达式为

$$P(d_m,w_n)=P(d_m)\sum_k P(z_k|d_m)P(w_n|z_k)$$

**似然函数**

这样的话，我们要做的事就是从文档集里估计出上面的参数。pLSA 是频率学派的方法，将模型参数看作具体值，而不是有先验的随机变量。所以，考虑最大化对数似然函数：

$$\begin{aligned}L(\theta)&=\ln \prod_{m=1}^M\prod_{n=1}^N P(d_m,w_n)^{n(d_m,w_n)}\\&=\sum_m\sum_n n(d_m,w_n)\ln P(d_m,w_n)\\&=\sum_m\sum_n n(d_m,w_n)(\ln P(d_m)+\ln P(w_n|d_m))\\&=\sum_m\sum_n n(d_m,w_n)\ln P(w_n|d_m)+\sum_m\sum_n n(d_m,w_n)\ln P(d_m)\end{aligned}$$

第二项可以直接去掉，那么不妨直接记：

$$\begin{aligned}L(\theta)&=\sum_m\sum_n n(d_m,w_n)\ln P(w_n|d_m)\\&=\sum_m\sum_n n(d_m,w_n)\ln \bigl[\sum_k P(z_k|d_m)P(w_n|z_k)\bigr]\end{aligned}$$

**参数估计：EM 算法迭代求解**

由于参数全部在求和号里被外层的 log 套住，所以很难直接求偏导数估计参数。到了这里，就轮到 EM 算法闪亮登场了。之前的一篇[简单介绍 EM 的博客](http://www.cnblogs.com/Determined22/p/5776791.html)里解决的是这样的问题：观测变量 y ，隐变量 z ，参数 $\theta$ ，目标函数 $L(\theta)=\ln P(y)=\ln \sum_k P(z_k)P(y|z_k)$ （省略 $\theta$ 依赖），Q 函数 $Q(\theta;\theta_t)=\mathbb E_{z|y;\theta_t}[\ln P(y,z)]=\sum_k P(z_k|y;\theta_t)\ln P(y,z)$ ，进而有 $\theta_{t+1}=\arg\max_{\theta}Q(\theta;\theta_t)$ 来迭代求取。

**1. E 步，求期望**

那么，仿照上面的写法，对于 pLSA 模型来说，Q 函数的形式为

$$\begin{aligned}Q(\theta;\theta_t)&=\sum_m\sum_n n(d_m,w_n) \mathbb E_{z_k|w_n,d_m;\theta_t}[\ln P(w_n,z_k|d_m)]\\&=\sum_m\sum_n n(d_m,w_n) \sum_k P(z_k|w_n,d_m;\theta_t)\ln P(w_n,z_k|d_m)\end{aligned}$$

(1) 联合概率 $P(w_n,z_k|d_m)$ 的求解：

$$P(w_n,z_k|d_m)=P(z_k|d_m)P(w_n|z_k, d_m)=P(z_k|d_m)P(w_n|z_k)$$

(2) $P(z_k|w_n,d_m;\theta_t)$ 的求解：

所谓的 $\theta_t$ 实际上就是上一步迭代的全部 $K\times M$ 个 $P(z_k|d_m)$ 和 $V\times K$ 个 $P(w_n|z_k)$ 。为了避免歧义，将时间步 $t$ 迭代得到的参数值加一个下标 $t$ 。

$$\begin{aligned}P(z_k|w_n,d_m;\theta_t)&=\frac{P_t(z_k,w_n,d_m)}{P_t(w_n,d_m)}=\frac{P_t(d_m)P_t(z_k|d_m)P_t(w_n|z_k)}{P_t(d_m)P_t(w_n|d_m)}\\&=\frac{P_t(z_k|d_m)P_t(w_n|z_k)}{P_t(w_n|d_m)}=\frac{P_t(z_k|d_m)P_t(w_n|z_k)}{\sum_j P_t(z_j|d_m)P_t(w_n|z_j)}\end{aligned}$$

基于以上两个结果，得到 Q 函数的形式为

$$\begin{aligned}Q(\theta;\theta_t)&=\sum_m\sum_n n(d_m,w_n) \sum_k P(z_k|w_n,d_m;\theta_t)(\ln P(z_k|d_m)+\ln P(w_n|z_k))\\&=\sum_m\sum_n n(d_m,w_n) \sum_k \frac{P_t(z_k|d_m)P_t(w_n|z_k)}{\sum\limits_j P_t(z_j|d_m)P_t(w_n|z_j)}(\ln P(z_k|d_m)+\ln P(w_n|z_k))\end{aligned}$$

终于，在这个形式里面，除了 $\theta$（全部 $K\times M$ 个 $P(z_k|d_m)$ 和 $V\times K$ 个 $P(w_n|z_k)$），已经全部为已知量。

**2. M 步，求极大值**

剩下的工作就是

$$\theta_{t+1}=\arg\max_{\theta}Q(\theta;\theta_t)$$

问题将会被概括为如下的约束最优化问题：

目标函数：$\max_{\theta}Q(\theta;\theta_t)$

约束：$\sum_n P(w_n|z_k)=1$，$\sum_k P(z_k|d_m)=1$

使用 Lagrange 乘数法，得到 Lagrange 函数为

$$Q(\theta;\theta_t)+\sum_k\tau_k(1-\sum_n P(w_n|z_k))+\sum_m\rho_m(1-\sum_k P(z_k|d_m))$$

令其对参数的偏导数等于零，得到 $K\times M+V\times K$ 个方程，这些方程的解就是最优化问题的解：

$$\frac{\partial}{\partial P(w_n|z_k)}=\frac{\sum\limits_m n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{ P(w_n|z_k)}-\tau_k=0,\quad 1\leq n\leq V, 1\leq k\leq K$$

$$\frac{\partial}{\partial P(z_k|d_m)}=\frac{\sum\limits_n n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{P(z_k|d_m)}-\rho_m=0,\quad 1\leq m\leq M, 1\leq k\leq K$$

方程的解为

$$P(w_n|z_k)=\frac{\sum_m n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\tau_k}$$

$$P(z_k|d_m)=\frac{\sum_n n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\rho_m}$$

注意到两个约束条件，即

$$\sum_n \frac{\sum_m n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\tau_k}=1$$

$$\sum_k \frac{\sum_n n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\rho_m}=1$$

从中可求得 $\tau_k$ 、$\rho_m$ ，所以方程的解为

$$P_{t+1}(w_n|z_k)=\frac{\sum_m n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\sum_n\sum_m n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}$$

$$P_{t+1}(z_k|d_m)=\frac{\sum_n n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}{\sum_k\sum_n n(d_m,w_n)P(z_k|w_n,d_m;\theta_t)}$$

当模型参数全部估计好后，便得到了完整的 pLSA 模型。上面的迭代过程很明显是一个频数估计（极大似然估计）的形式，意义很明确。模型使用 EM 算法进行参数估计时往往都会推导出这样的结果，例如 HMM。

**pLSA 模型小结**

上面的一系列公式因为总是在前面挂着两个求和号，所以看上去貌似挺热闹，其实它相比于朴素的三硬币模型的推导过程来说，无非就是多了作为条件出现的随机变量 d ，其余地方没有本质区别。

不难看出来，pLSA 模型需要估计的参数数量是和训练文档集的大小是有关系的 —— 因为有 $P(z_k|d_m)$ 。显然，在训练集上训练出来的这组参数无法应用于训练文档集以外的测试文档。

作为频率学派的模型，pLSA 模型中的参数被视作具体的值，也就谈不上什么先验。如果在真实应用场景中，我们对于参数事先有一些先验知识，那么就需要使用贝叶斯估计来做参数估计。

**LDA 模型**

相比于 pLSA ，2003 年提出的 LDA 模型显然名气更响，应用起来也丰富得多。LDA 将模型参数视作随机变量，将多项式分布的共轭先验（也就是 Dirichlet 分布）作为参数的先验分布，并使用 Gibbs sampling 方法对主题进行采样。中文资料简直不要太多，个人认为最经典的当属《 LDA 数学八卦》，作者将 LDA 模型用物理过程详细解释，抽丝剥茧地剖析了来龙去脉，看完之后会有一种大呼过瘾的感觉。英文资料推荐 Parameter estimatiom for text analysis。

![](https://images2017.cnblogs.com/blog/1008922/201707/1008922-20170729134102644-142192448.png)

LDA 模型。图来自 [5]

参考：

[1] Unsupervised Learning by Probabilistic Latent Semantic Analysis

[2] Latent Dirichlet Allocation, JMLR 2003

[3] [Topic Model Series [1]: pLSA](https://zhuanlan.zhihu.com/p/21811120)

[4] http://blog.tomtung.com/2011/10/plsa/#blei03

[5] Parameter estimatiom for text analysis

[6] LDA 数学八卦

[7] http://www.cnblogs.com/bentuwuying/p/6219970.html