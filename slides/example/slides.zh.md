<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01 · 第 01 讲</p>

# 词元化

<p class="subtitle">文本如何成为模型的输入？</p>

<p class="byline">Baojian Zhou<br>复旦大学 · 2026 年秋季</p>

Note:
这个简短的示例展示了课程讲义的格式，并不是第一节课的完整教案。
请学生谈谈，他们认为模型会如何表示一个中文句子。

---

<!-- .slide: id="representation" -->

## 文本表示

词元化器将文本映射为词元 ID 序列。

模型通过嵌入表读取这些 ID。

> 词元化器决定哪些文本片段共享同一个词表条目。

Note:
区分词元化器与使用它的模型。词元 ID 只有在其所属词元化器的词表中才有意义。

---

<!-- .slide: id="bytes" -->

## 码点与 UTF-8 字节

<div class="columns">
<div>
<h3>不同的度量方式</h3>
<p>一个 Unicode 码点标识一个字符值。</p>
<p>UTF-8 使用一个或多个字节对该值进行编码。</p>
</div>
<div>
<table>
<thead><tr><th>文本</th><th class="num">码点数</th><th class="num">字节数</th></tr></thead>
<tbody>
<tr><td><code>hello</code></td><td class="num">5</td><td class="num">5</td></tr>
<tr><td><code>你好</code></td><td class="num">2</td><td class="num">6</td></tr>
<tr><td><code>🙂</code></td><td class="num">1</td><td class="num">4</td></tr>
</tbody>
</table>
</div>
</div>

<p class="source">这些计数基于上方原样显示的字符串。一个可见符号可能包含多个码点。</p>

Note:
对于这些字符串，Python 的 len 计算码点数。JavaScript 的 string.length 计算 UTF-16 代码单元数，因此现场演示使用 Array.from。

---

<!-- .slide: id="code" -->

## 在 Python 中度量相同的文本

```python
examples = ["hello", "你好", "🙂"]

for text in examples:
    code_points = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    print(text, code_points, utf8_bytes)
```

在笔记本中运行该单元格之前，先预测每一行的输出。

Note:
笔记本中包含这个完全相同的示例。先暂停一下，再展示输出。

---

<!-- .slide: id="browser-demo" -->

## 探索这些计数

<form class="demo-form" id="utf8-demo">
<label for="demo-text">试试英文、中文或表情符号</label>
<input id="demo-text" type="text" value="你好🙂" maxlength="48" autocomplete="off" spellcheck="false">
<button type="button" id="demo-reset">重置示例</button>
</form>

<div class="demo-result" aria-live="polite">
<p><strong id="point-count">3</strong> 个 Unicode 码点</p>
<p><strong id="byte-count">10</strong> 个 UTF-8 字节</p>
</div>

<p class="caption">试试一个空格、一串数字或一个带重音符号的字母，并解释每次变化。</p>

Note:
这个演示完全在浏览器中运行，输入内容只保留在当前页面上。
对于分解形式的重音符号，请比较预组合字符 é 与后跟组合锐音符的 e。

---

<!-- .slide: class="section-slide" id="bpe" -->

<p class="eyebrow">一个小算法</p>

## 字节对编码

训练过程会反复合并一个高频相邻对。

Note:
介绍学习合并规则与应用已有合并列表之间的区别。
下一个示例从字符开始，以便清晰地展示计算过程。

---

<!-- .slide: id="worked-example" -->

## 一个玩具训练语料库

| 初始序列 | 频次 |
| :--- | ---: |
| `l o w` | 5 |
| `l o w e r` | 2 |

相邻对 `l o` 和 `o w` 都出现了 7 次。

<div class="fragment">
<p>选择 <code>l o</code> 来打破平局。下一次合并会组合 <code>lo w</code>。</p>
</div>

<p class="caption">在每个词内部统计相邻对，并用词频对计数加权。</p>

Note:
这是一个基于字符的玩具示例，没有词尾符号。请明确说明这些设定。
采用不同的平局处理方式，可能会学到不同的合并规则。

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">练习 E01 · 3 分钟 · 笔记本 E01</p>

## 哪个相邻对会最先合并？

| 初始序列 | 频次 |
| :--- | ---: |
| `a a a b` | 2 |
| `a b` | 1 |

统计每个相邻对，包括重叠出现的情况。

<div class="answer fragment">
<p><code>a a</code> 出现 4 次。<code>a b</code> 出现 3 次。</p>
</div>

Note:
请学生解释为什么在 a a a b 中，a a 会被计数两次。
相邻对频次可以包含重叠出现的情况。应用合并规则时从左到右替换互不重叠的出现位置。

---

<!-- .slide: id="results" -->

## 两次合并后的序列长度

| 阶段 | `low` 的序列 | `lower` 的序列 | 词元总数 |
| :--- | :--- | :--- | ---: |
| 初始 | `l o w` | `l o w e r` | 25 |
| 合并 `l o` | `lo w` | `lo w e r` | 18 |
| 合并 `lo w` | `low` | `low e r` | 11 |

<p class="caption">总数按 5 次 <code>low</code> 和 2 次 <code>lower</code> 加权。</p>

本例度量的是训练语料库上的序列长度。

Note:
验证总数分别为 5*3+2*5、5*2+2*4 和 5*1+2*3。
请思考：要在未见文本上评估词元化器，还需要哪些证据？

---

<!-- .slide: id="interactive-results" -->

## 两次合并：25 → 11 个词元

<div class="plot" data-plotly="assets/sequence-length.json" role="img" aria-label="Toy corpus total tokens decrease from 25 to 18 to 11 after zero, one, and two merges."></div>

<p class="caption">同一个玩具语料库：<code>low</code> × 5，<code>lower</code> × 2。悬停可查看精确计数。</p>

Note:
图表会从本地 Plotly 规范自动加载。它绘制的是上一张表中已经核对过的总数，而非模型基准测试结果。拖动可缩放；双击可重置。

---

<!-- .slide: id="equation" -->

## 平均序列长度

对于一个包含 $N$ 段文本的固定集合：

$$
\bar{L} = \frac{1}{N} \sum_{i=1}^{N} \left|\operatorname{encode}(x_i)\right|
$$

比较词元化器时，应使用相同的文本。

请报告词元化器的版本及所做的任何文本归一化处理。

Note:
这是每段文本的平均值，因此文档长度会影响它。请解释为什么比较时必须固定输入文本集合。

---

<!-- .slide: class="references" id="references" -->

## 参考文献

- [Jurafsky 和 Martin，《词语与词元》](https://web.stanford.edu/~jurafsky/slp3/2.pdf)<br>阅读第 2.3–2.4 节：Unicode 与 BPE（2026 年 8 月草稿）。
- [Sennrich、Haddow 和 Birch（2016）](https://aclanthology.org/P16-1162/)<br>阅读第 3.2 节，了解基于 BPE 的子词方法。
- [Python Unicode 指南](https://docs.python.org/3/howto/unicode.html)<br>复习 Unicode 字符串与 UTF-8 编码。

Note:
可通过页面顶部的“笔记本”链接访问笔记本。
