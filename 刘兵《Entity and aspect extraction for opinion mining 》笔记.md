# 刘兵《Entity and aspect extraction for opinion mining 》笔记

> 转载请声明出处。

这是一本书的一个章节（49页），书名叫《Data mining and knowledge discovery for big data》2014年Springer出版。

## Introduce
介绍了一些Opinion Mining的背景，这里不说。

survey book: 
1. Pang and Lee(2008)
2. Liu(2012)

三种粒度：篇章级、句子级、方面级

篇章级：篇章级情感分类可能是最广泛的研究问题。
句子级：对文档的单个句子进行情感分类，但不是每个句子都包含意见的。所以第一个任务就是判断句子是否包含意见，被称为『主观性分类』。
方面级：篇章级和句子级虽然有许多应用场景，但是一个被分为正向的句子中，并非所有aspect都是正向的。所以需要细化到aspect。

"Aspect-based opinion mining"第一次在Hu and Liu(2014)被提出，当时叫"Feature-based opinion mining"。
它的基本任务是提取和概况人们表达的实体和方面的意见，包含三个核心子任务：
1. 识别和提取实体
2. 识别和提取实体的方面
3. 计算实体和实体方面的情感倾向

"I brought a Sony camera yesterday, and its picutre quality is great." 它的asepct为picture quality，实体为Sony camera。
本章节针对这两个任务进行展开。
一些研究者用feature和object表达aspect和entity，也有一些研究者不区分aspect和entity，直接看作opinion target。

## Aspect-based Opinion Mining Model

### Model Concepts

#### Defintion: entity
一个entity可以是产品、服务、事件、组织或者话题。它关联着一个pair, e:(T,W): T为组件(components(or parts))的层级结构， W为e的属性(attribute)。每个component或者sub-component也有它自己的属性>

例子：entity iPhone 有一系列component（如battery和screen）和一系列attribute（如voice quality、size和weight）battery组件也有它自己的属性（如bettery life和battery size）

该定义可以表达为一棵树。

#### Definition: aspect and aspect expression

在实战中，简化该定义经常是有效的，因为nlp很难，学习层级结构更难。所以，我们简化和摧毁树结构到两级结构，用aspects来表达components和attributes。在简化的树中，根节点为entity，二阶节点为aspect。

aspect expression 是一个在文本中出现的实际单词或短语。
它经常为名词或名词短语，但也有动词、动词短语、形容词、副词。
我们把句子中的以名词或名词短语形式出现的aspect expression成为explicit aspect expression。其他形式就成为implicit aspect expressions。implicit较为复杂。

#### Definition: entity expression
entity expression是出现在文本中指示一个特定entity的实际单词或短语。

#### Defintion: opinion holder
表达意见的人或组织，经常被称为opinion sources

#### Definition: opinion
opinion有两个主要类型：regualr opinions和comparative opinions(Liu, 2010;Liu,2012)

五元组：
$(e_i,a_{ij},oo_{ijkl},h_k,t_l)$

当一个意见描述entity整体，一般aspect用GENERAL表达。


#### Model of entity

entity $e_i$可以用整体和一个有限的aspect集合$A_i=\{a_{i1},a_{i2},…,a_{in}\}$表达。
entity可以用一个entity expression集合来表示，$OE_i=\{oe_{i1},oe_{i2},…,oe_{is}\}$
每个aspect $a_{ij} \in A_{id}$可以用一个aspect expression集合表示，$AE_{ij}=\{ae_{ij1},ae_{ij2},…,ae_{ijm}\}$

#### Model of opinionated document

一篇包含意见的文档d包含来自意见持有者集合$\{h_1,h_2,…,h_p\}$的关于实体集合$\{e_1,e_2,…,e_r\}$的意见。
每个实体$e_i$的意见可以用entity本身和aspects $A_{id}$表达。

#### Objective of opinion mining
目标是挖掘Document中的五元组

### Aspect-based Opinion Summary
略

## Aspect Extration

aspect extaction和entity extration都归属于信息抽取，目标都是从无结构文本中自动抽取结构化信息。但是传统的信息抽取技术经常是应用于正式文本（新闻、论文等），对于opinion mining application就有困难。我们目标是从包含意见的文档中抽取细粒度的信息（reviews, blogs and forum discussions），其中包含着大量的噪音和有着独特的特征。所以，设计针对于opinion document的抽取算法是有必要的。

当前的研究主要基于在线评论，通常有两种格式：
1. Pros, Cons and the detailed review，如某些手机网站
2. Free format

本文主要针对格式2。

### Extraction Approaches

这里只介绍近年(2014)aspect抽取的主要方法。
正如前面所说的，aspect有两种类型：explicit和implicit。我们先讨论explicit。
我们把现有的提取方法分类三个主要类型：
1. language rule
2. sequence models
3. topic models

#### Exploiting Language Rules

基于语言规则的系统在信息抽取领域有着很长的使用历史。这些规则基于上下文模式，获取文本中一个或多个terms的不同特性或特性。在评论中，我们使用aspects和opinion word或其他词语之间的语法关系来推导提取规则。

**Hu and Liu(2014)**第一个提取使用关联规则来提取aspects，主要步骤：
1. 找出频繁名词和名词短语作为frequent aspects
2. 使用aspects和opinion words的关系来识别非频繁的aspect。

使用frequent名词和名词短语作为aspect简单有效

**Blair-Goldensohn et al.(2008)**通过考虑有情感的句子中的名词短语或指示情感的一些句法模式来改进算法。几个过滤方法被应用来移除不像的aspect，例如，移除那些附近没有已知情感词的aspect。
基于frequency的思路后来也被应用。**(Popescu and Etzioni, 2005; Ku et al., 2006; Moghaddam and Ester, 2010; Zhu et al., 2009; Long et al., 2010).**

用改进的opinion word和aspect关系来提取aspect的思路可以被归为使用依存关系。
**Zhuang et al.(2006)**使用依存关系来从影评中提取aspect-opinion pairs。
**Wu et al.(2009)**用了一个短语依存句法分析工具来提取名词短语和动词短语作为aspect候选。与一般的依存句法分析工具不同，短语依存句法分析工具识别短语的依存关系。**Kessler and Nicolov(2009)**也用了依存关系。

**Wang and Wang(2008)**提出了一个同时识别aspect和opinion word的方法。给定种子opinion words，用bootstrapping的方法来交替识别aspect和opinion word。互信息（mutual information）被应用于衡量潜在aspect和opinion word的关联程度。另外，语言规则被用于识别非频繁的aspects和opinion words.类似的bootstrapping思路也在**Hai et al.(2012)**提出。

Double propagation(**Qiu et al.,2011**)进一步发展了前面的思路。像**Wang and Wang(2008)**，该方法只需要一个初始的种子opinion words。它观察到意见几乎都是有target的，而且句子中的aspect和opinion word有自然的关系，因为opinion word用来修饰target。此外，它发现opinion words之间有关系，aspects也是。所以，opinion words可以通过已识别的aspect来识别，aspect也可以通过已识别的opinion word来识别。已抽取的opinion word和aspect可以用来识别新的opinion word和aspect。这个传播过程执行到不在有新的opinion word和aspect被发现。因为这个过程包含opinion word和aspect，所以叫double propagation。抽取规则根据opinion word和aspect之间的不同关系来设计。

Double Propagation方法在中等大小的语料中有效，但是对于大的或者小的语料，它可能会造成低precision和低recall。原因是基于直接依存关系的规则在语料中有很大几率引入噪音，而对于小语料来说，规则太局限。为了克服这些缺点，**Zhang et al.(2010)**扩展了double propagation。它包括两步：aspect extraction和aspect ranking。对于aspect extraction，依然使用double propagation。但是，引入了一些新的语言模式（e.g.,part-whole关系规则）。提取之后，它将候选aspect根据重要性排序，考虑两个主要因素：aspect candidate和aspect frequency。前者描述了一个候选aspect多像一个真实的aspect，有三个线索：第一个就是aspect经常被多个opinion word修饰；第二个是aspect可以用多个part-whole规则提取，比如，在car领域，"the engine fo the car"和"the car has a big engine"，我们推断"engine"是car的一个aspect；第三个是aspect可以用opinion word修饰关系、part-whole关系和其他语言规则联合提取。如果一个aspect不仅被opinion word修饰，而且通过part-whole提取，我们可以推断他是一个有着high confidence的真实aspect，比如"there is a bad hole in the mattress"，它强烈地指示了"hole"是mattress的一个aspect，因为他被"bad"修饰和在part-whole关系里。此外，在opinion words、linguistic pattern和aspect之间有一些互相加强的关系。如果一个形容词修饰多个真实aspect，它就很可能是一个good opinion word。类似地，如果一个候选aspect通过许多opinion words和linguistic pattern提取出来，它就很可能是一个真实aspect。所以**Zhang et al.**用HITS算法(Klernberg, 1999)来衡量aspect relevance。Aspect Frequency是影响aspect ranking另一个重要因素。

**Liu et al.(2012)**也利用了opinion word和aspect的关系来提取。但是他们把aspect和opinion word之间的opinion relation identification看作是词对齐任务（word alignment）。他们用基于词的翻译模型(Brown et al.,1993)来实现单语词对齐。基本上，aspect和opinion word的关联用翻译概率来衡量，能比语言规则更准确更有效地获取aspect和opinion word之间的opinion relations。

**Li et al.,(2012a)**提出了一个领域自适应的方法来抽取跨领域的aspect和opinion word。在一些情况下，目标领域没有标注数据，但源领域有大量标注数据。基本的思路就是利用源领域抽取的知识来帮助识别目标领域的aspect和opinion word。该方法包括两个步骤：（1）识别一些共同的opinion words作为种子，然后从源领域中提取高质量的opinion aspect种子。（2）一个叫"Relational Adaptive bootstrapping"的bootstrapping方法用来扩展这些种子。首先，通过在源领域的标注数据和目标领域的新标注数据来迭代训练一个跨领域的分类器，然后用它来预测目标未标注数据的label。第二，top预测的aspect和opinion word被挑选来作为候选。第三，利用之前迭代中提取的句法规则来构建一个aspect和opinion word之间的二部图。使用基于图的得分计算算法获取top候选，分别加入到aspect和opinion word list中。

除了利用aspect和opinion word的关系以外，**Popescu and Etzioni(2005)**提出了利用一个上下文中的鉴别关系来提取aspects的方法，也就是aspects和产品class的关系。他们首先提取频繁的名词短语作为候选aspect，然后使用候选和一些产品class的部分整体关系鉴别器(meronymy discriminators)之间的PMI评估每个候选词。例如"scanner"类别的meronymy discriminators是像"of scanner", "scanner has","scanner comes with"等模式。PMI公式
$$PMI(a,d)={hits(a\land d)\over hits(a)hits(d)}$$
a是候选aspect，d为meronymy discriminators。通过搜索引擎实现hits()。
该算法也用WordNet的is-a层次结构和形态结构线索从attribute中区别components/parts。

**Kobayashi et al.(2007)**提出了一个从blog中提取aspect-evaluation和aspect-of关系的方法，它利用了aspect, opinion expression和product class的关联关系。例如，在aspect-evaluation pair提取中，evaluation expression首先由词典决定。然后，句法关系被用来找出它对应的aspect来生成候选pair。这些候选pairs通过一个由结合上下文和统计线索这两种信息训练得到的分类器来测试和验证。上下文线索为句子中词的句法关系，它可以由依存语法决定；统计学线索是标注的aspect和evaluations的共现。


#### Squence Models

主要是Hidden Markov Model和Conditional Random Fields。有监督学习。

##### Hidden Markov Model

**Jin et al.(2009a and 2009b)**利用词汇化的HMM来从评论中抽取product aspects和opinion expression。与传统HMM不同，他们将如POS和词汇模式融入到HMM中。例如，一个观察变量用pair($word_i$,$POS(word_i)$)表示。

##### Conditional Random Fields

**Jakob and Gurevych(2010)**利用CRF从包含意见的句子中抽取opinion target(or aspects)。他们用Token, POS, Short Dependency Path, Word Distance作为特征输入。使用Inside-Outside-Begin(IOB)标注方案。

**Li et al.,2010a**做了类似的工作。为了能对句子级中的长距离的用连接词("and", "or", "but")连接的依存关系，以及aspect，positive opinion和negative opinion之间的深层依存句法建模，他们使用了skip-tree CRF模型来发现产品aspect和opinoin。


#### Topic Model

主题模型在NLP和文本挖掘中被广泛运用，它基于文档的多个主题分布和每个主题的词分布。一个主题模型是文章的生成模型(generative model)。通常，它指定文章的生成过程。具体看《LDA数学八卦》。

主题模型可以用于aspect抽取。我们可以认为每个aspect是一个元语言模型，即词语的多项分布。虽然这样的表示很难解析为aspect，但是它的优势就是表达一样或相近aspect的不同词语可以被自动地聚到一起。如今，用主题模型抽取aspect有着大量的研究。他们基本上是吸收和扩展了pLSA(Hofmann, 2011)和LDA模型(Blei et al., 2003)。

##### Probabilistic Latent Semantic Analysis

原理请阅读《LDA数学八卦》。

对于aspect抽取任务，我们可以把产品aspect当做opinion document中的潜在topic。**Lu et al.(2009)**提取了在短文本中发现aspect和聚类的方法。他们假设每条评论都可以被解析成为格式为<head term, modifier>的opinion phrase，和利用head term和modifiers的共现信息将这个opinion phrase融入pLSA模型。通常，head term是一个aspect，modifier是opinion word。提出的方法定义k元语言模型：$\Theta = (\theta_1,\theta_2,…,\theta_k)$作为k主题模型，每一个都是head terms的多项分布。注意每个modifier都可以用一个header term的集合表示，表示公式为：
$$d(w_m)=\{w_h|(w_m,w_h)\in T\}$$
$w_h$表示head term，$w_m$表示modifier。

实际上，一个modifier可以被当做一个混合模型的一个sample。
$$p_{d(w_m)}(w_h)=\sum_{j=1}^k[\pi_{d(w_m),j}p(w_h|\theta_j)]$$
$\pi_{d(w_m),j}$是第j个aspect的特定modifier的混合权重(modifier-specific mixing weight)，加起来等于1。modifiers$V_m$集合的对数似然值(log-likelihood)为
$$log\ p(V_m|\Delta)=\sum_{w_m\in v_m}\sum_{w_h\in v_h}\{c(w_h,d(w_m))\times log\ \sum_{j=1}^k[\pi_{d(w_m),j}p(w_h|\theta_j)]\}$$
$c(w_h,d(w_m))$为head term$w_h$和modifiers$w_m$的共现次数，$\Delta$为所有模型参数集合。

利用EM算法，k主题模型可以被估计，aspect expression可以被聚合。另外，**Lu et al.**使用了共轭先验融入人类知识来制定aspect的聚类。因为提出的方法对head terms和modifier的共现建模，所以他可以利用更多有意义的句法关系。

**Moghaddam and Ester(2011)**通过加入对评论的潜在排序信息到模型来提取aspect和他们的对应排序，扩展了以上pLSA模型。

但是pLSA方法的主要缺点就是它是内在转换，即没有直接的方法把已学习的模型应用到新文档。在pLSA中，集合中的每个文档d用一个混合系数$\theta$来表示，但是它并不对集合以外的文档进行定义。

##### Latent Dirichlet Allocation(LDA)

基本的LDA模型请阅读《LDA数学八卦》

基于LDA的模型在几个研究中被用于aspect抽取。**Titov and McDonald(2008a)**指出全局的主题模型(像pLSA和LDA)可能不适合发现aspect。pLSA和LDA都用了文档的词袋模型表示，它依赖于主题的分布差异和词语的共现来识别每个主题中的topic-word概率分布。但是，对于opinion文章(如review)来说，他们非常不同。也就是，每个文档都是讨论相同的aspect，这使得全局主题模型(global topic model)效率低和只对entities发现有效(如品牌和产品名称)。为了解决这个问题，他们提取了多粒度的LDA(MG-LDA)来发现aspect，它对global topic和local topic这两个不同类型的主题进行建模。像在pLSA和LDA中，对于一篇文章的global topic分布是固定的。但是，local topic的分布则允许不同。一个文档中的一个词是要么从global topic的多项分布，要么从这个词的local context特定的local topic的多项分布。它假设aspect会被local topic获取，global topic会获取评价item的属性。例如，一条London hotel的评论："…public transport in London is straightforward, the tube station is about an 8 minute walk… or you can get a bus for \$1.50"。这条评论可以当做是global topic *London* (words:"London","tube","\$")和local topic(aspect) *location* (words:"transport", "walk", "bus").

MG-LDA能区分local topics。但是由于local topics和ratable aspects之间的many-to-one映射，这个对应是不明显的。它缺乏topics到aspects的直接分配。为了解决这个问题，**Titov and McDonald(2008b)**扩展了MG-LDA模型和构建了一个文本和aspect rating的联合模型，叫做Multi-Aspect Sentiment model(MAS)。它包含两个部分：第一个部分是基于MG-LDA来构建代表ratable aspect的topics；第二部分是针对每个aspect的一系列分类器(sentiment predictors),它在aspect-specific rating的帮助下推断local topics和aspect的映射。他们的目标是利用rating信息来识别更多aspect。

LDA的思想也被应用和扩展在(**Branavan et al.,2008; Lin and He, 2009; Brody and Elhadad, 2010; Zhao et al., 2010; Wang et al., 2010; Jo and Oh, 2011; Sauper et al., 2011; Moghaddam and Ester, 2011; Mukajeee and Liu, 2012**)。Branavan利用*Format 1*的评论格式的关键词来协助提取aspect。关键词是基于分布的和正字的(orthographic)属性来聚类，隐topic model应用于review文本。然后，一个最终的图模型将他们两个结合。**Lin and He(2009)**提出了一个join topic-sentiment model(JST)，它通过加入一个sentiment层来扩展了LDA。它能从文本中同时发现aspect和sentiment。**Brody and Elhadad(2010)**提出了用local版本的LDA来识别aspect，它作用于句子而非文档，利用了小量的直接对应于aspect的topics。**Zhao et al.(2010)**提出了一个MaxEnt-LDA混合模型来联合发现aspect words和aspect-specific opinion words，它能利用句法特征来帮助区分aspects和opinion words。**Wang et al.(2010)**提出了一个回归模型基于学习了的潜在aspects来推断aspect ratings和aspect weights。**Jo and Oh(2010)**提出了一个*Aspect and Sentiment Unification Model(ASUM)*来对面向不同asepct的sentiment建模。**Sauper et al.(2010)**提出一个联合模型，它只工作于已经从reviews中提取的小片段，联合了HMM和topic modeling，其中HMM拟合了词类型序列(aspect, opinion word, or background word)。**Moghaddam and Ester(2011)**提出了一个叫ILDA的模型，它基于LDA和加入了潜在aspect和rating建模。ILDA能看做一个生成过程：首先生成一个aspect，随后生成它的rating。特别地，对于生成每个opinion phrase，ILDA首先从LDA模型中生成aspect$a_m$，最后。一个head term$t_m$和一个sentiment$s_m$从$a_m$和$r_m$的条件分布中生成。**Mukajeee and Liu(2012)**提出了两个模型(SAS and ME_SAS)来使用种子对aspect和aspect specific sentiments联合建模，从而从语料发现aspects。种子反映了用户对发现特定aspects的需求。

其他关于topic model相关工作有topic-sentiment model(TSM)。**Mei et al.(2007)**提出这个模型来对在blog中的topic和sentiment联合建模，它用了一个positive sentiment model和一个negative sentiment model附加在aspect模型上。他们在文章级别进行情感分析而不是在aspect级别。在(**Su et al., 2008**)中，作者也提出了一个基于mutual reforcement方法的聚类算法来识别aspect。类似的工作有(**Scaffidi et al., 2007**)，他们提出了一个针对于产品aspect的语言模型，它假设产品aspect在产品review文本中比在通用英文文本中更频繁提到。但是，当语料规模小的时候，统计是不可靠的。

总的来说，主题建模是一个强大和灵活的建模工具。它也在概念上和在数学上都很优秀。但是，它只适合找出一些general/rough的aspects，难以找到细粒度的或者准确的aspects。我们认为它过于以统计为中心，有局限。如果我们往自然语言和知识中心转移，提出更平衡的方法，可能会有更多成果。

#### Miscellaneous Methods

**Yi et al.(2003)**提出基于likelihood-ratio test的方法提出aspect。**Bloom et al.(2007)**人工构建了aspects的分类，指示aspect类型。他们也通过review的一个样本来构建aspect词典，他们人工检验这些种子词典，用WordNet来挖掘额外的词语。**Lu et al.(2010)**利用Freebase来获取一个topic的aspects，用它们来组织零散的意见，生成一个结构化的意见摘要。**Ma and Wan(2010)**利用Centering theory(**Grosz et al.1995**)来从新闻评论中提取评价对象。**Ghani et al.(2006)**把aspect抽取当成分类问题，用了传统的监督学习方法和半监督学习方法来抽取产品aspects。**Yu et al.(2011)**使用一个叫one-class SVM的部分监督方法来提取aspects，只需要标注一些正例(是aspect的例子)。他们只从Pros和Cons抽取aspects。**Li et al.(2012b)**把抽取aspect当做浅层语义解析问题。每个句子构建一棵解析树，其中的结构化的句法信息用来识别aspect。

#### Aspect Grouping and Hierarchy

人名通常会使用不同的词语和表达来描述同一个aspect。例如，*photo*和*picture*在数码相机领域中表达同一个aspect。虽然topic model可以识别和聚合aspect，但是结果并不是细粒度的，因为这样的模型是基于词共现而不是语义。所以，一个topic往往是关于一个general topic的相关词list，而不是表示同一个aspect的词list。例如，一个topic关于*battery*可能包含像*life,battery,charger,long,short*等词语。我们可以清晰地看到，这些词语并不代表同一个东西，虽然他们可能经常共现。我们可以先提取aspect expression，然后把他们聚合到不同的aspect catergories。

聚合指示同一个aspect的aspect expression对opinion应用来说是很关键的。虽然WordNet和其他词典可以帮助这个任务，但是他们由于很多同义词都是领域独立的，所以经常无效。例如，*picture*和*movie*是moview评论的同义词，但是他们在数码相机领域不是同义词，因为*picture*更接近*photo*而*movie*更接近*video*。注意到虽然一个aspect的大部分aspect expressions是领域同义词，但是他们不总是同义词。例如，*expensive*和*cheap*都可以指示*price*这个aspect，但他们不是*price*的同义词。

**Liu, Hu and Cheng(2005)**试图用WordNet同义词集来解决这个问题，但是结果不令人满意，因为WordNet对解决领域独立的同义词方面不够有效。**Carenini et al.(2005)**也提出了一个方法解决这个问题。他们的方法基于使用字符串相似度、同义词和距离衡量来定义的几种相似度矩阵。但是，它要求事先给定一个分类。这个算法合并每一个发现的aspect expression到分类中的一个aspect结点。

**Guo et al.(2009)**提出了一个多层次的潜在语义关联技术（叫mLSA）来聚合产品aspect expression。在第一层，aspect expression的所有词都通过使用LDA被聚合到一个concepts/topics集合中。这个结果用来构建一些潜在topic结构。在第二层，aspect expression通过LDA根据他们的潜在topic结构和上下文片段来被聚合。

**Zhai et al.(2010)**提出了一个半监督方法来将aspect expression聚合到用户自定义的aspect group或category中。每个group代表一个特定的aspect。为了反映用户的需求，他们首先给每个group人工标注一小部分种子。这个系统然后使用基于标注的种子和未标注的样本来将剩余的aspect expression分配到合适的group。这个方法使用了Expectation-Maximization(EM)算法。两块先验知识被使用来为EM提供更好的信息，也就是：(1)共用一些相同的词语的aspect expression更可能属于同一个aspect group;(2)在词典中属于同义词的aspect expression更可能属于同一个aspect group。**Zhai et al.(2011)**进一步提出了一个无监督方法，不需要事先标注样本。此外，它进一步通过词典相似度加强。这个算法也利用了一些自然语言知识来提取更有区分性的分布上下文来帮助聚合。

**Mauge et al.(2012)**使用基于聚类算法的最大熵来聚合aspect。它首先训练一个最大熵分类器来决定两个aspect是同义词的概率。然后，一个无向有权图构建出来。每个节点代表一个aspect。每条边权重代表两个节点的概率。最后，近似图分割方法(approximate graph partitioning method)用来聚合aspect。

与aspect聚合相关的aspect层级用来将产品aspect表示成一棵树或层级。根节点是实体名称。每个非根节点是一个entity的组件或子组件。每一个link都是*part-of*关系。每一个结点关联一系列的aspect。**Yu et al.(2011b)**提出了一个方法来创建aspect层级。这个方法从一个初始层级开始，一个个地插入aspect直到所有aspect被分配。每个aspect通过语义距离学习来插入到最佳位置。**Wei and Gulla(2010)**学习基于aspect hierarchy trees的情感分析。

#### Aspect Ranking

#### Mapping Implicit Aspect Expressions

有许多隐形aspect expression类型。形容词可能是最经常出现的类型。许多形容词修饰或描述一些特定的entity属性。例如，形容词*heavy*通常描述entity的*weight*。*Beautiful*一般用来描述entity的*look*或者*appearance*。也不是说这些形容词只描述这些aspects。他们准确的意思可以是领域独立的。例如，*heavy*在句子*the traffic is heavy*并不描述交通的*weight*。注意一些隐性aspect expression很难提取和映射，例如，*fit in pockets*在句子*This phone will not easily fit in pockets*。

将隐性aspect映射到他们的显性aspect的研究并不多。在**Su et al.(2008)**中，聚类算法被用来映射隐性aspect expression，这些aspect expression被假设为情感词，对应着显性aspect。这个方法利用了显性aspect和情感词之间的相互增强关系来生成一个共现pair。这样的一个pair可能指示着情感词描述aspect，或者aspect关联着情感词。这个算法通过将显性aspect集和和情感词集合分别迭代聚类来挖掘映射关系。在每一词迭代中，在对一个集合聚类之前，使用其他集合的聚类结果来提升集合的pair相似度。集合中的pair相似度由集合内相似度和集合间相似度的线性组合来决定。两项在集合内的相似度是传统的相似度，在集合间的相似度基于aspect和情感词的关联程度来计算。关联程度(或mutual reinforcement relationship)由一个二分图建模。如果一个aspect和opinion word在句子中共现，那么他们是相连的。这些链接也基于共现频数来确定权重。在迭代聚类之后，强连接的aspect和情感词group生成最后的映射。

在**Hai et al.(2011)**中，一个两阶段共现关联规则挖掘方法被提出来匹配隐性aspect(被假设为情感词)的显性aspect。在第一阶段，这个方法生成关联规则，将语料中频繁在句子中共现的pair中的每个情感词作为condition，显性aspect作为consequents。在第二阶段，对consequents(显性aspect)聚类来为每个规则中的情感词生成更加鲁棒的规则。为了应用或测试，给定没有显性aspect的情感词，找出最好的规则簇，然后分配这个簇中的代表性词语作为最后识别的aspect。

**Fei et al.(2012)**聚焦于找到被意见形容词(opinion adjectives)指示的隐性aspect(主要是名词)，例如，为形容词*expensive*识别*price*、*cost*等。他们提出了一个基于词典的方法，尝试从形容词词典中识别出属性名词。他们把问题定义为集合分类问题(colletive classification problem)，它可以利用词语的词典关系(如同义词、反义词、下位词和上位词)来分类。

一些其他相关工作包含在(**Wang and Wang,2008;Yue et al.,2011b**)。


#### Identifying Aspects that Imply Opinions

**Zhang and Liu(2011a)**发现在一些指示产品的领域名词和名词短语中aspect可能隐含着opinion。在许多案例中，这些名词不是主观的而是客观的。他们包含的句子也是客观性的句子，但是暗含着正向或者负向的opinion。例如，床褥评论中一个句子*"Within a month, a vally formed in the middle of the mattress."*。这里*valley*指示着床褥的质量，也暗含着负向的opinion。识别这样的aspect和他们的极性是一项非常具有挑战性但是在意见挖掘中非常有用的工作。

**Zhang and Liu**观察到对于含有暗含opinion的一个产品aspect来说，并没有直接修饰它的opinion word，或者修饰它的opinion word有着相同的意见倾向。

**Observation：**没有opinion word直接修饰被评价的产品aspect(*"valley"*)：
*"Within a month, a vally formed in the middle of the mattress."*

**Observation：**有opinion形容词修饰被评价的产品aspect(*"valley"*)：
*"Within a month, a bad vally formed in the middle of the mattress."*
这里，形容词*bad*修饰*valley*。它不像另一个句子中的正向opinion word也修饰*valley*，如，*"good valley"*。所以，如果一个产品aspect被正向和负向opinion形容都修饰的话，它不太可能是一个被评价的产品aspect。

基于这些观察，他们设计了如下两个步骤来识别暗含正向或负向意见的名词产品aspect：
**Step 1:**候选词识别(Candidate Identification)：这一步决定了每个名词aspect附近的情感上下文。这个直觉是如果一个aspect出现在负向(或正向)的意见上下文中比出现在正向(或负向)上下文更加频繁，我们可以推断它的极性是负向的(或正向的)。一个统计测试(总体比例测试)被用来测试它的显著性。这一步生成一个正向意见的候选aspect列表和一个负向意见的候选aspect列表。
**Step 2:**剪枝(Pruning)：这一步对两个列表进行剪枝。思路是当一个名词产品aspect被正向和负向opinion word都直接修饰时，它不太可能是被评价的产品aspect。


#### Identifying Resource Noun

**Lin(2010)**指出存在一些词或短语类型本身没有情感，但是当他们出现在一些特定的上下文中，它暗含着正向或负向的意见。在情感分析可以到达下一个准确率层次之前，所有这些表达必须要被提取和相关问题必须要被解决。

```
1. Postive <- consume no or little resource
2.          | consume less resource
3. Negative <- consume a large quantity of resource
4.          |  consume more resource
         
Figure 6: 包含资源的表述的情感倾向。
```

这样的一种表述类型包含了资源，这种情况经常出现在许多应用领域中。例如，*money*在几乎所有领域中是一种资源，*ink*在printer领域中是一种资源，*gas*在car领域中是一种资源。如果一个设备消耗了大量资源，它是不令人满意的(negative)。如果一个设备消耗极少资源，他是令人满意的(positive)。例如，句子*"This laptop needs a lot of battery power"*和句子*"This car eats a lot of gas"*分别在laptop领域和car领域中暗含着负向的情感。这里*gas*和*battery power*都是资源，我们把这些词语成为资源项(*resource terms*，包括词语和短语)。他们是一种特殊的产品aspect。

在包含资源的情感方面，Figure 6中的规则可用(**Liu, 2010**)。规则1和规则3代表了包含资源和暗含情感的常态句子，而规则2和规则4代表了包含资源和暗含情感的比较句式句子，例如，*"this washer uses much less water than my old GE washer"*。

**Zhang and Liu(2011a)**把问题定义为二分图问题，并提出了一个迭代算法来解决问题。这个算法基于如下观察：
**Observation:**句子中关于资源使用的情感或意见表达经常由如下三元组决定：
$$(verb, quantifier, noun\_term)$$
其中，*noun_term*是代表资源的一个名词或名词短语。

这个方法使用这样的三元组来帮助在领域语料中识别资源。模型使用了基于二分图的循环定义来反映*资源使用动词(resource usage verbs，consume*)和*资源项(如，water*)之间特定的增强关系。量词(quantifier)不用在计算，而用在识别候选动词和资源项。这个算法假设给定一个量词列表(不多，可人工构建)。基于循环定义，这个问题通过使用像HITS算法(**Kleinberg, 1999**)这样的迭代算法来解决。为了启动迭代计算，一些全局的*种子资源(seed resources)*被用来发现和评分一些健壮的*资源使用动词*。这些得分然后被应用到任意应用领域的迭代计算的初始化。当算法收敛时，一个排序过的候选资源项列表被识别出来。



