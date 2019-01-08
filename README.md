## Sequence Match with CompareAggregate Model

* 预处理：

    * 只保留汉字，用 jieba 分词，用 gensim 包对训练集训练 word2vec 模型
    * 部分 ans 过长，发现长度大于 57 的句子不到 1%，因此预处理时进行截断

* 模型

    ```mermaid
    graph TB;
    	in_q--dropout-->drop_q
    	in_a--dropout-->drop_a
    	drop_q--fc_sigmoid-->q1
    	drop_q--fc_tanh-->q2
    	q1-->v_q(mul_q)
    	q2-->v_q
    	drop_a--fc_sigmoid-->a1
    	drop_a--fc_tanh-->a2
    	a1-->v_a(mul_a)
    	a2-->v_a
    	subgraph Compare
    	v_q-->e_qa
    	v_a-->e_qa
    	e_qa--softmax_a_axis-->w_q
    	e_qa--softmax_q_axis-->w_a
    	w_q-->h_q
    	v_a-->h_q
    	w_a-->h_a
    	v_q-->h_a
    	v_a-->t_a
    	h_a-->t_a
    	v_q-->t_q
    	h_q-->t_q
    	end
    	subgraph Aggregate
    	t_q-->t0( )
    	t_q-->t1( )
    	t_q--convs-->t2( )
    	t_q-->t3( )
    	t_q-->t4( )
    	t0-->conv_q
    	t1-->conv_q
    	t2--maxpool-->conv_q
    	t3-->conv_q
    	t4-->conv_q
    	t_a-->t10( )
    	t_a-->t11( )
    	t_a--convs-->t12( )
    	t_a-->t13( )
    	t_a-->t14( )
    	t10-->conv_a
    	t11-->conv_a
    	t12--maxpool-->conv_a
    	t13-->conv_a
    	t14-->conv_a
    	conv_q-->cat_qa(cat_qa)
    	conv_a-->cat_qa
    	end
    	cat_qa--fc-->out
    ```

    （github 上的 repo 大多只使用了一条 Compare 的分支，并且在 attention 时把 softmax 的 dim 写反了）

* 注

    使用预训练好的word2vec模型，效果更好。