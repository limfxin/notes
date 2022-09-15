# EVENTSKG: 
## 概述
- 论文名称：A Knowledge Graph Representation forTop-Prestigious Computer Science Events Metadata
- [论文地址](https://www.researchgate.net/publication/326890754_EVENTSKG_A_Knowledge_Graph_Representation_for_Top-Prestigious_Computer_Science_Events_Metadata_10th_International_Conference_ICCCI_2018_Bristol_UK_September_5-7_2018_Proceedings_Part_I)
- 数据查询语言 
1. 六个主题：
   1. information systems (IS)
   2. security and privacy (SEC)
   3. artificial intelligence (AI)
   4. computer systems organization (CSO)
   5. software and its engineering (SE)
   6. web (WWW)
-  [URL](http://purl.org/events_ds)
- [github](https://github.com/saidfathalla/EVENTS-Dataset)
- 数据构建的过程
-  [数据展示](https://saidfathalla.github.io/EVENTS-Dataset/EVENTS.html)
## 时序知识图谱的定义和结构
## 关于时序时间图谱的定义
![](../attachment/39a21c5f1aaa01b19b3ace397c84cf4.png)
可以看到时序知识图谱由节点（entities）和时序关系构成（temporal relations)组成，其中节点由现实世界节点(realworld entities)和事件(events）组成。
![](../attachment/409B0627-20E1-4568-9C72-268742821FF9.jpeg)
时序知识图谱中的节点由两个信息来描述：第一个是$e_eur$表示该节点的独特的指示，第二个是$e_time$表示这个现实节点的存在时间或者是时间的发生时间
![](../attachment/872B4AC3-C0DB-4395-A33C-1BCB785AB390.jpeg)
第三个定义是关于边的关系
## 时序知识图谱的逻辑结构

![](../attachment/68F5A0C1-ABDA-4E99-B5E1-25AB6AD3FAAF.jpeg)
1. EVENTkg是在[SEM](SEM.md)的基础上进行建构的，SEM的结构和缺点参考SEM文档，在上图中绿色的是从SEM中继承的，黄色部分是在EVENTkg中附加的
2. sem:event sem:actor sem:place 都为sem:core的三个子类，用来表征一个节点的事件，参与者和地点信息
3. 在上图中，sem:core和enentKg-s:Relation中间的rdf:subject 和ref:object用来表示两个节点的关系，在图中链接到一个sem:core当中，其实是不同的
4. enentKg-s:Relation 中的links和mentions属性分别表示一个实体的连接数和提及数，可以用来计算关系强度(relation strength)和流行程度（event popularity metrics）
![](../attachment/E1BE6276-7A22-4FAC-BC55-03F22E1DF0FD.jpeg)
>上图为在逻辑结构图中具体资源的网站


![](../attachment/1C4DB102-B3ED-4ED8-BBA8-0A9D97783CFC.jpeg)
> 上图就是EVTNTkg的一个实例，表示的含义是奥巴马的第二次当选，这个含义当中就包括了两个节点，一个是现实世界的实体节点，一个是事件节点，中间是关系，用上文所说的rdf:subject和rds:object进行连接，根据定义，在圆角长方体的中的indentifier，连接的长方体是节点的属性，包括名称和起始时间等。



## EVENTkg的生成

![](../attachment/3dfc59cf72560d984955ade1568cfbe.png)
> 数据的加工代码见之前的链接
1. input and pre-processing（数据输入和预处理）
	1. 数据来源
		- Wikidata
		- YAGO
		- DBpedia
		- Wikipedia Current  Events Portal
	2. 使用语言 EN, FR, DE, RU and PT
	3. 预处理的过程：
		1. 确定术语（Terms）：（确定关键词maybe）
		2. 抽取数据表达式（Date expressions）：比如日期等
		3. 定义表示事件关系的谓词映射（Mapping of predicates representing event relations）也就是把找到的术语和数据和eventkg中的对应起来。注意：这里只是定义这张表格，并没有具体抽取。
		![](../attachment/ffd8bf83a87fddfaeb40f0fb53eec43.png)
>上图即为对应关系		
2. indentification an extraction of events（事件的识别和提取）

3. extraction of event and entities of relations
	1. 提取数据的有效时间
	2. 提取间接关系
	3. 根据上面的关系对应的表格提取实体事件关系
	4. 关系强度和流行性分析
4. integration（整合）
	1. 创建了一个命名图eventKG-g:event_kg来储存integration 和fusion的结果
	2. 从不同的信息源 获取 owl:sameAs 联系，不同语言，不同来源的联系是不同的
	3. 合并整合联系相同的节点
5. fusing
	1. 时间融合
	2. 地点融合
	3. 类型融合
6. output
	1. 最后结果以RDF 形成呈现
## 生成示例
![](../attachment/64f60d87ca323f193f509b22d79b68e.png)
> 上图是抽取出来的原始数据

![](../attachment/2422891e247fdaf628bf65a16f5483d.png)
> 经过extraction

![](../attachment/93f4bd7431bc04d778385db42be1a60.png)
>经过integration and fusion


