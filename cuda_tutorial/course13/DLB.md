  DLB: Decoupled Look-Back "解耦回看"
    是一种做并行prefix scan / prefix sum的方法,
    尽量像串行前缀和那样只读一遍输入, 写一遍输出, 接近单遍扫描

  以V4为例
    GPU麻烦的点在于, block和block之间有依赖
      1. 每个block先扫自己的tile, 写出block sum
      2. 再对block sum进行一次san
      3. 把block sum加回原输出

  1. prefix_sum本身
    x = [3, 1,  7,  0,  4,  1,  6,  3,  2,  5,  2,  6,  1,  3,  4,  2]
    y = [3, 4, 11, 11, 15, 16, 22, 25, 27, 32, 34, 40, 41, 44, 48, 50]
    CPU_串行
    template <typename T>
    void prefix_sum_CPU(T* in, T* out, int32_t len) {
      for (int i = 0; i < len; ++i) {
        out[i] = in[i];
      }
      for (int i = 1; i < len; ++i) {
        out[i] = out[i - 1] + out[i];
      }
    }

  2. 分块之后问题出在哪?
    按4个元素分块
    chunk0 = [3, 1, 7, 0]
    chunk1 = [4, 1, 6, 3]
    chunk2 = [2, 5, 2, 6]
    chunk3 = [1, 3, 4, 2]

    每个chunk内部可以先独立做局部前缀和
    chunk0 local scan = [3, 4, 11, 11], aggregate0 = 11
    chunk1 local scan = [4, 5, 11, 14], aggregate1 = 14
    chunk2 local scan = [2, 7,  9, 15], aggregate2 = 15
    chunk3 local scan = [1, 4,  8, 10], aggregate3 = 10

    最终结果
    chunk0 = chunk0 = [3, 4, 11, 11]
    chunk1 = chunk1 + aggregate0 = [15, 16, 22, 25]
    chunk2 = chunk2 + aggregate0 + aggregate1 = [27, 32, 34, 40]
    chunk3 = chunk3 + aggregate0 + aggregate1 + aggregate2 = [41, 44, 48, 50]
    所以分块的问题在于: 每个chunk怎么知道自己前面所有chunk的总和, 也就是自己的全局offset

  3. 最朴素的方法: 两阶段/三阶段
    1. 每个chunk自己做局部scan
      aggregates: [11, 14, 15, 10]
    2. 对aggregate进行prefix_sum(exclusive chunk prefix)
      aggregates: [11, 25, 40, 50]
      offset: [0, 11, 25, 40]
    3. offset加回每个chunk
    缺点:
      1. 先把所有chunk的aggregates收集起来
      2. 单独再scan一遍aggregates
      3. 回头给每个chunk补上offset
      多kernel/单kernel(全局barrier), 多次global往返

  4. 另一种更直觉但不够好: 链式等待
    chunk0做完后, 得到prefix = 11
    chunk1等chunk0, 拿到11, 算出自己的prefix = 25
    chunk2等chunk1, 拿到25, 算出自己的prefix = 40
    chunk3等chunk2, 拿到40, 算出自己的prefix = 50
    缺点: 依赖性过强

  5. DLB
    我不一定非要等前一个chunk的完成prefix算出来, 我只要能从前面若干chunk的描述信息里,
    把自己的offset凑出来就行
    5.1 decoupled
      解耦:
        chunk内部局部scan
              和
        chunk之间全局offset的传播
        不再强绑定
      当前chunk可以先把块内活干掉, 不必一上来就卡死在"前驱chunk还没给我prefix"
    5.2 look-back
      当前chunk会往前看前面的chunk状态
      前面每个chunk会留下一个"描述符", 里面通常至少有:
      aggregate: 这个chunk的总和
      inclusive_prefix: 到这个chunk结尾为止的全局前缀
      status: 当前处于什么状态

  6. 用CPU真正走一遍DLB
    chunk内scan已经结束, 等待offset
    chunk0 aggregate = 11
    chunk1 aggregate = 14
    chunk2 aggregate = 15
    chunk3 aggregate = 10
    6.1 chunk0
      它前面没有任何chunk
        exclusive_prefix(chunk0) = 0
        inclusive_prefix(chunk0) = exclusive_prefix(chunk0) + aggregate0 = 11
        chunk0 = [3, 4, 11, 11]
      发布描述符
        desc[0] = {
          aggregate = 11,
          inclusive_prefix = 11,
          status = PREFIX_READY
        }
    6.2 chunk1
      他往前看chunk0,
        如果chunk0的PREFIX_READY出来了
          exclusive_prefix(chunk1) = inclusive_prefix(chunk0) = 11
          inclusive_prefix(chunk1) = exclusive_prefix(chunk1) + aggregate1 = 25
          chunk1 = [4,5,11,14] + 11 = [15,16,22,25]
      发布描述符
        desc[1] = {
          aggregate = 14,
          inclusive_prefix = 25,
          status = PREFIX_READY
        }
    6.3 chunk2(真正体现DLB的地方)
      假设一个时序
        chunk0全部做完, PREFIX_READY
        chunk1只做完了局部scan, 发布了aggregate = 14
        但是chunk1的完整inclusive_prefix = 25, 还没发布
        如果链式, 只能等待(强依赖)
        DLB
          1. chunk1
            发现
                desc[1].status = AGGREGATE_READY
                desc[1].aggregate = 14
            那就先把这个aggregate = 14加入running total
            carry = 14
          2. chunk0
            发现
                inclusive_prefix(chunk0) = 11,
                status(chunk0) = PREFIX_READY
          3. 求解
            exclusive_prefix(chunk2) = inclusive_prefix(chunk1)(还没发布)
                inclusive_prefix(chunk1) = exclusive_prefix(chunk1) + aggregate1
                exclusive_prefix(chunk1) = inclusive_prefix(chunk0)
            exclusive_prefix = inclusive_prefix(chunk0) + carry
          虽然chunk1的完整prefix还没出来(inclusive_prefix(chunk1): 未知)
          但是chunk2已经知道自己的offset, 且立刻发布
            inclusive_prefix(chunk2) = exclusive_prefix(chunk2) + aggregate2 = 40
      chunk2不必等待chunk1的完整prefix, 只要chunk1的aggregate已经出来,
        并且更早某个chunk的完整prefix可用, 它就能把自己的offset算出来
    6.4 chunk3
      6.4.1 假设一个时序
        chunk0: PREFIX_READY, inclusive_prefix(chunk0) = 11
        chunk1: AGGREGATE_READY,  aggregate1 = 14
        chunk2: AGGREGATE_READY,  aggregate2 = 15
        往前看, 看chunk2, aggregate2 = 15
        看chun1, aggregate1 = 14
        看chunk0, inclusive_prefix(chunk0) = 11
        exclusive_prefix(chunk3) = inclusive_prefix(chunk2)(还没发布)
          inclusive_prefix(chunk2) = exclusive_prefix(chunk2) + aggregate2
          exclusive_prefix(chunk2) = inclusive_prefix(chunk1)(还没发布)
          inclusive_prefix(chunk1) = exclusive_prefix(chunk1) + aggregate1
          exclusive_prefix(chunk1) = exclusive_prefix(chunk0)
        exclusive_prefix(chunk3) = inclusive_prefix(chunk0) + aggregate1 + aggregate2
      6.4.2 假设一个时序
        chunk0: PREFIX_READY, inclusive_prefix(chunk0) = 11
        chunk1: PREFIX_READY, inclusive_prefix(chunk1) = 25
        chunk2: AGGREGATE_READY, aggregate2 = 15
        往前看, 看chunk2, aggregate2 = 15
        看chun1, inclusive_prefix(chunk1) = 25
        exclusive_prefix(chunk3) = inclusive_prefix(chunk2)(还没发布)
          inclusive_prefix(chunk2) = exclusive_prefix(chunk2) + aggregate2
          exclusive_prefix(chunk2) = inclusive_prefix(chunk1)
        exclusive_prefix(chunk3) = inclusive_prefix(chunk1) + aggregate2
  总结: 某些前驱至少已经公布aggregate, 更早有一个chunk(任意一个)已经公布完整prefix(inclusive_prefix)

  7. 为什么说它是"有界冗余工作"
    chunk3在求offset时, 可能会重新读取chunk2的aggregate、chunk1的aggregate;
      而chunk4之后也可能再次读到这些东西(允许少量回看)
