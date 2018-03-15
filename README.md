使用TensorFlow的基本步骤
===

# 学习任务
学习使用TensorFlow，并以california的1990年的人口普查中的城市街区的房屋价值中位数作为预测目标，使用均方根误差（RMSE）评估模型的准确率,并通过调整超参数提高模型的准确率

## 设置
加载必要的库+数据导入以及一些简单的处理

加载必要库

``` python
import math

from IPython import display//display模块可以决定显示的内容以何种格式显示
from matplotlib import cm  //matplotlib为python的2D绘图库
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np        //numpy为python的科学计算包，提供了许多高级的数值编程工具
import pandas as pd       //pandas是基于numpy的数据分析包，是为了解决数据分析任务而创建的
from sklearn import metrics//sklearn是一个机器学习算法库
import tensorflow as tf   //tensorflow是谷歌的机器学习框架
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10//为了观察数据方便，最多只显示10行数据
pd.options.display.float_format = '{:.1f}'.format
```
加载数据集
``` python
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
```
为了防止数据集中出现病态排序，先对数据进行随机化处理，此外并将`median_house_value` 调整为以千为单位，这样更符合现实生活中的习惯，并且模型就能够以常用范围内的学习速率较为轻松地学习这些数据。
``` python
//california_housing_dataframe.index原始序列集索引
//np.random.permutation（）随机打乱原索引顺序
//california_housing_dataframe.reindex（）以新的索引顺序重新分配索引
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```
|  | longitude | latitude | housing_median_age | total_rooms | total_bedrooms | population | households | median_income | median_house_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14836 | -122.2 | 37.5 | 26.0 | 1777.0 | 555.0 | 1966.0 | 497.0 | 3.0 | 211.0 |
| 13475 | -122.0 | 37.1 | 21.0 | 2387.0 | 357.0 | 913.0 | 341.0 | 7.7 | 397.7 |
| 3391 | -117.9 | 33.7 | 27.0 | 1596.0 | 297.0 | 1703.0 | 289.0 | 4.1 | 184.9 |
| 4108 | -118.0 | 33.8 | 34.0 | 1038.0 | 175.0 | 578.0 | 174.0 | 4.9 | 200.0 |
| 1901 | -117.3 | 32.7 | 44.0 | 1934.0 | 325.0 | 783.0 | 316.0 | 4.9 | 358.6 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7731 | -118.4 | 34.0 | 44.0 | 1913.0 | 441.0 | 1295.0 | 432.0 | 4.0 | 266.4 |
| 4008 | -118.0 | 34.1 | 20.0 | 2063.0 | 496.0 | 1573.0 | 468.0 | 3.2 | 157.1 |
| 1612 | -117.2 | 33.6 | 6.0 | 13724.0 | 2269.0 | 5860.0 | 1986.0 | 4.0 | 183.0 |
| 6593 | -118.3 | 34.0 | 46.0 | 1098.0 | 426.0 | 1510.0 | 374.0 | 2.1 | 156.3 |
| 9219 | -119.1 | 34.4 | 52.0 | 1409.0 | 359.0 | 981.0 | 304.0 | 2.8 | 199.3 |
17000 rows × 9 columns

## 检查数据
目的是为了在使用之前对数据有一个初步的了结
``` python
california_housing_dataframe.describe()//输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数
```
longitude | latitude | housing_median_age | total_rooms | total_bedrooms | population | households | median_income | median_house_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 17000.0 | 17000.0 | 17000.0 | 17000.0 | 17000.0 | 17000.0 | 17000.0 | 17000.0 | 17000.0 |
| mean | -119.6 | 35.6 | 28.6 | 2643.7 | 539.4 | 1429.6 | 501.2 | 3.9 | 207.3 |
| std | 2.0 | 2.1 | 12.6 | 2179.9 | 421.5 | 1147.9 | 384.5 | 1.9 | 116.0 |
| min | -124.3 | 32.5 | 1.0 | 2.0 | 1.0 | 3.0 | 1.0 | 0.5 | 15.0 |
| 25% | -121.8 | 33.9 | 18.0 | 1462.0 | 297.0 | 790.0 | 282.0 | 2.6 | 119.4 |
| 50% | -118.5 | 34.2 | 29.0 | 2127.0 | 434.0 | 1167.0 | 409.0 | 3.5 | 180.4 |
| 75% | -118.0 | 37.7 | 37.0 | 3151.2 | 648.2 | 1721.0 | 605.2 | 4.8 | 265.0 |
| max | -114.3 | 42.0 | 52.0 | 37937.0 | 6445.0 | 35682.0 | 6082.0 | 15.0 | 500.0 |

## 构建第一个模型
本次练习，我们将尝试预测 `median_house_value`（每个街区的房屋价值的中位数），它将是我们的标签（有时也称为目标target）。我们将使用 `total_rooms` (每个街区的房间总数)作为输入特征。

为了训练模型，我们将使用 TensorFlow Estimator(一种评估器) API 提供的 LinearRegressor 接口。此 API 负责处理大量低级别模型搭建工作，并会提供执行模型训练、评估和推理的便利方法。
>estimator（估计量）[统计学中](https://en.wikipedia.org/wiki/Statistics "统计")，**estimator**是基于观测数据计算给定量的估计值的规则，即通过给定的一些example，通过一定的规则，计算给出预测值。
在tensoflow中其是`tf.Estimator` 类的一个实例，用于封装负责构建 TensorFlow 图并运行 TensorFlow 会话的逻辑，是处于最顶层的面向对象的高级api

>LinearRegressor 线性回归，可以理解为 通过训练得出一条逼近example的线

| 工具包 | 说明 |
| --- | --- |
| Estimator (tf.estimator) | 高级 OOP API。 |
| tf.layers/tf.losses/tf.metrics | 用于常见模型组件的库。 |
| TensorFlow | 低级 API |

以下是构建模型的步骤
### 第 1 步：定义特征并配置特征列
* * *
 为了将我们的训练数据导入 TensorFlow，我们需要指定每个特征包含的数据类型。在本练习及今后的练习中，我们主要会使用以下两类数据：
*   **分类数据**：一种文字数据。在本练习中，我们的住房数据集不包含任何分类特征，但您可能会看到的示例包括家居风格以及房地产广告词。
*   **数值数据**：一种数字（整数或浮点数）数据以及您希望视为数字的数据。有时您可能会希望将数值数据（例如邮政编码）视为分类数据（我们将在稍后的部分对此进行详细说明）。
>当我们有了example以后，其通常包含许多特征，比如本次例子中的housing_median_age ， total_rooms等，在之后为了更好的处理对应特征的对应数据，我们选择先给这些特征分个类，而经过前辈的验证发现主要是使用**分类数据**和**数值数据**，比如人的性别这个特征可以看成分类数据，男女在对枪战游戏的喜爱程度上，就得分开站队了，所以这时候，这样得分类数据很有意义，而像考试某门考试分数这个特征还是数值数据更合理啦

在 TensorFlow 中，我们使用一种称为“**特征列**”的结构来表示特征的数据类型。特征列仅存储对特征数据的描述；不包含特征数据本身。
>意义以及解释在代码注释中

一开始，我们只使用一个数值输入特征 `total_rooms`。以下代码会从 `california_housing_dataframe` 中提取 `total_rooms` 数据，并使用 `numeric_column` 定义特征列，这样会将其数据指定为数值： 
* * *
```
# Define the input feature: total_rooms.
# 取数据集中得'total_rooms'这一列作为输入特征
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
# 将一个名叫"total_rooms"的特征列定义为**数值数据** ，这样的定义结果存在feature_columns中
# 即上文所说得**特征列**中，这时候特征列其实只是一个存储了分类信息的集合，具体使用的时候需要
# 特征集合和特征列结合起来，分类器才能识别的呢
feature_columns = [tf.feature_column.numeric_column("total_rooms")] 
```
* * *
 **注意**：`total_rooms` 数据的形状是一维数组（每个街区的房间总数列表）。这是 `numeric_column` 的默认形状，因此我们不必将其作为参数传递。 
* * *

### 第 2 步：定义目标
 接下来，我们将定义目标，也就是 `median_house_value`。同样，我们可以从 `california_housing_dataframe` 中提取它： 

* * *
```
# Define the label.
# 将"median_house_value"列的数据从数据集中取出作为target，这就是我们搭建的模型所要学习的东# 西
targets  =  california_housing_dataframe["median_house_value"]
```
### 第 3 步：配置 LinearRegressor
 接下来，我们将使用 LinearRegressor 配置线性回归模型，并使用 `GradientDescentOptimizer`（它会实现小批量随机梯度下降法 (SGD)）训练该模型。`learning_rate`参数可控制梯度步长的大小。
***
* **梯度** (gradient):**偏导数**相对于所有自变量的向量。在机器学习中，梯度是模型函数偏导数的向量。梯度指向最速上升的方向。
* **梯度下降法** (gradient descent):一种通过计算并且减小梯度将**损失**降至最低的技术，它以训练数据为条件，来计算损失相对于模型参数的梯度。通俗来说，梯度下降法以迭代方式调整参数，逐渐找到**权重**和偏差的最佳组合，从而将损失降至最低。
***
上面的术语解释读起来比较抽象，并且为了防止文章篇幅过长，看了上面忘了下面，所以模型的实现过程放在遇到问题在下面解释

*  下图显示了机器学习算法用于训练模型的迭代试错过程
![图片](https://i.loli.net/2018/03/15/5aaa0b9767214.png)
输入特征=>模型预测=>根据结果计算一下损失（损失就是距离target的差距），然后将参数更新，再放回模型中预测，直至收敛，使得损失变得最小，这时候的参数就是我们想要的参数
*  下图是上图中"计算参数更新"的绿框中的内容
![图片](https://i.loli.net/2018/03/15/5aaa0e58c36ba.png)
假设我们能够将所有种可能情况全都计算一遍，那么得到的一定是一个类似于这样的碗状图，在其中必定有一点是损失最低的点，但是现实种我们肯定不会有那么大的计算能力和时间去计算出每个结果，我们通常采用一种叫做**梯度下降法**的方式来"快速"的找到损失最低的点（梯度下降法属于一种优化算法，虽然并不是最好的优化算法，但是其方式简单，应用也很多）。
*  **起点**是随意选定的，因为在预测的开始时，没有人知道权重（w1,w2,w3..b）该是什么，可以设置为0，也可以设置为1，无所谓。通过模型一次计算，计算得出损失（这时候损失并不重要，肯定极大，没有参考意义），然后计算起点处的**偏导数**（如果只有一个权重那就是导数了），得出起点处的偏导数，而**梯度**是偏导数的矢量（即包含了此处偏导数的**方向**和**大小**），可以想象一下抛物线y=ax²+bx+c  在x0处的导数，其大小的绝对值是随着x0的值而变化的，并且有正负之分，绝对值大小代表**大小**，正负代表**方向**，所以依据**梯度**就可以确定权重值调节的方向。
*  至此，调节的基本原理说的就差不多了，那么剩下的问题就是如何更好的优化，以便用最少的计算量最快的速度去达到目的。
***
* **学习速率**（也称为步长）
1.学习速率过慢
![](https://i.loli.net/2018/03/15/5aaa19ac52484.png)
2.学习速率过快
![](https://i.loli.net/2018/03/15/5aaa19f6a5a9e.png)
3.学习速率比较好的
![](https://i.loli.net/2018/03/15/5aaa1a4e9a1c6.png)

如果让其按照每个点本身的**梯度**大小来调节权值，那实在是太慢了，所以我们可以为其乘上一个学习速率，意如其名，这样可以人手动的调节学习速率（或许有的人会担心，当即将逼近损失最小的点时，这样会不会不太准确了，放心好了，我们并不需要那么的准确的权值，99%和98%的区别不是太大，但是所要付出的计算量却是超大的）

附上谷歌提供的：[优化学习速率](https://developers.google.cn/machine-learning/crash-course/fitter/graph)体验

下面是两种个效果更好的**梯度下降算法方案**，第二种更优
**随机梯度下降法** (**SGD**) ：它每次迭代只使用一个样本（批量大小为 1）。“随机”这一术语表示构成各个批量的一个样本都是随机选择的。(假设有10000个样本，每次从中随机选一个来执行梯度下降)

**小批量随机梯度下降法**（**小批量 SGD**）是介于全批量迭代与 SGD 之间的折衷方案。小批量通常包含 10-1000 个随机选择的样本。小批量 SGD 可以减少 SGD 中的杂乱样本数量，但仍然比全批量更高效。（每次随机选一批）

**注意**：为了安全起见，我们还会通过 `clip_gradients_by_norm` 将**梯度裁剪**应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。 

解释完毕，以上
***
华丽分割线
* * *
```
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 这里的clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种
# 比较常用的梯度规约的方式，解释起来太费事啦。。。。略略
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# Configure the linear regression model with our feature columns and optimizer
# Set a learning rate of 0.0000001 for Gradient Descent.
# 线性回归模型，tf.estimator.LinearRegressor是tf.estimator.Estimator的子类
# 传入参数为**特征**列和刚才配置的**优化器**，至此线性回归模型就配置的差不多啦
# 前期需要配置模型，所以是与具体数据（特征值，目标值是无关的）
linear_regressor = tf.estimator.LinearRegressor(
 feature_columns=feature_columns,
 optimizer=my_optimizer
)
```
### 第 4 步：定义输入函数
要将加利福尼亚州住房数据导入 `LinearRegressor`（刚刚配置好的线性回归模型），我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。（看不明白接着往下看）

我们在输入如数据之前，得将数据先处理好（按照大小啊，数据的特性啊），就像之前给数据分类为**数值**的还是**分类**的一样，因为要使用**小批量随机梯度下降法**，所以数据还需要按固定大小分批一下子
***
首先，我们将 _Pandas_ 特征数据转换成 NumPy 数组字典。然后，我们可以使用 TensorFlow Dataset API根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 `batch_size` 的多批数据，以按照指定周期数 (num_epochs) 进行重复。
>不知道你还能不能记得一开始导入包时候的代码注释
>>import numpy as np        //numpy是python的科学计算包，提供了许多高级的数值编程工具
	import pandas as pd        //pandas是基于numpy的数据分析包，是为了解决数据分析任务而创建的

这里的大概过程就相当于使用_Pandas_的转换工具将我们从California的住房集种抽出来的数据来一个格式转换，目的是为了让接下来的数据更好更容易的被处理，比如炸薯条的话，得先给土豆削皮，然后就得切条了，对于刀工不好的人，应该是挺希望这时候的土豆是一个标准的长方体的吧，这样切起来很爽很舒服，在这里的格式转换就是这个目的，土豆还是土豆。

下面是对即将使用的函数的参数的说明
***
**注意**：如果将默认值 `num_epochs=None` 传递到 `repeat()`，输入数据会无限期重复。

然后，如果 `shuffle` 设置为 `True`，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。`buffer_size` 参数会指定 `shuffle` 将从中随机抽样的数据集的大小。
```
# 自定义个输入函数
# 输入的参数分别为 
# features:特征值（房间数量）
# targets: 目标值（房屋价格中位数）
# batch_size:每次处理训练的样本数（这里设置为1）
# shuffle: 如果 `shuffle` 设置为 `True`，则我们会对数据进行随机处理
# num_epochs:将默认值 `num_epochs=None` 传递到 `repeat()`，输入数据会无限期重复

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	# dict(features).items():将输入的特征值转换为dictinary（python的一种数据类型，
    # lalala = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com'}）
	# 通过for语句遍历，得到其所有的一一对应的值（key：value）
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
    # Dataset.from_tensor_slices（(features,targets)）将输入的两个参数拼接组合起来，
	# 形成一组一组的**切片**张量{（房间数，价格），（房间数，价格）....}
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```

最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
### 第 5 步：训练模型
### 第 6 步：评估模型

## 调整模型超参数




 







