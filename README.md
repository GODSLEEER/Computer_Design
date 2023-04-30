# 1. 基于随机森林算法实现的二手帆船价格预估模型及价格趋势分析

- [1.基于随机森林算法实现的二手帆船价格预估模型及价格趋势分析](#1-基于随机森林算法实现的二手帆船价格预估模型及价格趋势分析)
  - [1.1. 项目说明](#11-项目说明)
  - [1.2. 资源](#12-资源)
    - [1.2.1. 本文件夹的文件结构](#121-本文件夹的文件结构)
    - [1.2.2. “Result List”文件夹的文件结构](#122-Result List文件夹的文件结构)
    - [1.2.3.[“figs”文件夹的文件结构](#123-figs文件夹的文件结构)
  - [1.3. 步骤](#13-步骤)
  - [1.4. 算法](#14-算法)
    - [1.4.1. 问题情景](#141-问题情景)
    - [1.4.2. 算法思路](#142-算法思路)
    - [1.4.3. 时间复杂度](#143-时间复杂度)
    - [1.4.4. 算法优缺点](#144-算法优缺点)
      - [1.4.5.1. 优点](#1451-优点)
      - [1.4.5.2. 缺点](#1452-缺点)
  - [1.5. 数据集](#15-数据集)

## 1.1. 项目说明

此代码是中国大学生计算机设计大赛的参赛作品，受 GPL-3.0-only 开源协议保护。开源地址：https://gitee.com/GODSLEEPER/computer_-design

项目代码在python 3.75 环境中编辑并测试通过。

## 1.2. 资源

### 1.2.1. 本文件夹的文件结构

| 名称或后缀         | 包含文件数 | 作用                                     |
| ------------------ | ---------- | ---------------------------------------- |
| `README.md`        | 1          | 本文档                                   |
| `*.py`             | 7          | 实现本算法的核心代码、测试文件等         |
| `*.xlsx`           | 3          | 收集的数据集                             |
| `*.csv`            | 6          | 收集的数据集                             |
| `*.model`          | 1          | 用于存储模型结构及参数的文件             |
| `requirements.txt` | 1          | 用于快速配置运行本项目的python环境的文件 |
| `LICENSE`          | 1          | GPL-3.0-only 开源协议                    |
| `Result List`      | 10         | 代码产生的结果文件夹                     |
| `figs`             | 10         | 代码绘制的图像文件夹                     |

### 1.2.2. “Result List”文件夹的文件结构

| 名称                         | 类型     | 作用                                                |
| ---------------------------- | -------- | --------------------------------------------------- |
| `Score.png`                  | PNG 图片 | 展示了模型训练得到的分数及模型的pearson系数、RMSE等 |
| `ImportanceVariable4.png`    | PNG 图片 | 展示了地区效应的影响度                              |
| `ParameterResult_ML.xlsx`    | XLSX表格 | 展示了所得到的随机森林模型的参数结构                |
| `PredictionError.png`        | PNG图片  | 展示了模型在数据集上的误差分布                      |
| `Spearmanr_Rigion_List.xlsx` | XLSX表格 | 展示了模型对各个地区二手船进行预测的Spearman系数    |
| `tree.svg`                   | SVG图片  | 使用浏览器可流畅打开，展示了随机森林模型的结构      |
| `True And Predictions.png`   | PNG图片  | 展示了预测值和真实值的分布                          |
| `VariableImportance.png`     | PNG图片  | 展示了模型各个输入变量的重要性程度                  |

### 1.2.3. “figs”文件夹的文件结构

| 名称                      | 类型     | 作用                                           |
| ------------------------- | -------- | ---------------------------------------------- |
| `savefig_example*.png`    | PNG 图片 | 为模型预测各个种类二手船的平均百分比误差       |
| `spearmanr_Hongkong1.png` | PNG 图片 | 绘制了地区效应对双体船和单体船各自的价格影响力 |

## 1.3. 使用

### 1.3.1. 步骤

##### 1.python环境配置

​	在项目根目录下打开终端或命令行窗口，执行如下命令：

​	pip install -r requirements.txt

​	而由于使用了相对路径，因此我们需要保证Final.csv、Final.xlsx、make and variant.xlsx

、place and economic.xlsx、dikaer.xlsx、Hong Kong.xlsx文件均在命令行所在目录下。

##### 2.二手船价格预估模型的训练

​	在本项目中存在已经训练好的模型文件Regreesion_Price_Predict.model，将其放在命令行所在目录下即可使预测、精度评估等代码正常运行。

​	如果希望自己训练一个二手船价格预估模型，我们有两种方案：一是直接使用Regression_Tree_Train.py代码直接训练得到model模型文件，二是可以使用Regression_Tree_Prameter_Finder.py程序搜索模型训练最佳参数值，在Regression_Tree_Train.py更改模型训练参数值完成训练。

​	Regression_Tree_Train.py同时还具有初步展示模型训练效果、绘制随机森林结构为dot文件、存储模型参数表（ParameterResult_ML.xlsx）、分析变量重要性程度的功能。

##### 3.二手船价格的估计和精度评估

​	运行Variant_Error.py可以测试模型对各个二手船种类评估的百分比误差，运行Variant_Precision.py程序可以测试模型对各个种类二手船评估的R-Squared。

##### 4.对二手船价格估计模型的进一步分析

​	运行Variant_Predict_Spearman_Economy.py可以通过模型分析地区经济因素对不同种类二手船只的价格影响；运行Variant_Predict_Spearman_Catamarans.py可以通过模型分析香港地区效应对单体船和双体船各自的影响程度。

### 1.4.1. 问题情景

​	开发本作品的目的是为了帮助人们更快速，准确地估算二手游艇的价值，以便进行买卖或投资决策。我们的程序面向的用户为二手游艇的买家和卖家，以及对游艇投资感兴趣的人群。主要功能包括对训练数据的属性进行权重排序，让用户只需输入游艇信息，程序就能输出对游艇的估价。我们的程序将对全球各地的二手游艇进行数据采集和整理，并通过机器学习技术提高算法精度和效率，该算法还具有一定的可扩展性，以支持更多的游艇类型和更多的船体数据，让用户能够得到更加准确的估价。

### 1.4.2. 算法思路

​	本算法 = 随机森林算法 + R-squared, pearson 模型评估+ 变量重要性分析+one-hot编码+ spearman相关度分析 

### 1.4.4. 时间复杂度

​	对于一个决策树来说，n个样本，m个特征，时间复杂度为O(mn logn)。如果是用M个树来投票，时间复杂度为O(M(mn log n))。计算过程中需要随机选取部分特征(subset of the feature)，需要额外的时间去处理这个过程，所以可能需要更多时间。

### 1.4.5. 算法优缺点

#### 1.4.5.1. 优点

* 强大的鲁棒性。我们的模型基于回归树和随机森林技术，这是成熟且经过广泛检验的经典ML方法。我们的模型是具有鲁棒性的，尤其是在面对多个复杂数据集时，并且保持其准确性。这意味着，如果我们有一艘帆船及其所在地区的精确数据，无论其他微小因素如何变化，该模型都可以对其上市价格做出可靠的预测。由于随机森林的引入，该模型具有抗过度拟合的能力。
* 普遍性。我们建立的模型是基于帆船甚至汽车贸易中的一些共同属性，如产品的生产日期和地区发展状况。这意味着，该模型还能够处理其他未出现在本项目中的帆船，甚至其他车辆（如私人飞机），只需进行一些修改。
* 只管。该模型是基于常识开发的，因此对于理解来说是直观的。我们引入的或建模中的所有变量都直接来自现实数据，或者只是进行了一些小的转换，比如我们用一个one-hot编码替换了Make。因此，在实际生产环境中部署该模型应该很简单。

#### 1.4.5.2. 缺点

* 缺乏对大型数据集的能力：尽管随机森林算法已经足够快了，但当随机森林中有大量决策树时，训练过程中所需的空间和时间将非常大，这将导致模型速度变慢。
* 易受噪声影响：随机森林已被证明在一些噪声较多的问题或回归问题上过度拟合。因此，我们需要在开始训练之前对数据集进行预处理，这对于灵活部署来说代价较高。

## 1.5. 数据集

网上并无现成的数据集，需要自己制作数据集。本文件夹提供了一个数据集，即Final.xlsx以及Final.csv文件，集合了三千四百余条包括二手船物理信息、品牌信息、生产地经济信息的数据，这是本项目的核心数据集，其他表格文件大都是这个文件的子集。

| 文件名称                | 所包含数据                                                   |
| ----------------------- | :----------------------------------------------------------- |
| dikaer.xlsx             | 统一生产时间后，集合每一种二手船变体与所有地区及其经济数据的数据集 |
| Final.*                 | 核心数据集，包含船只数据和经济数据                           |
| Hong Kong.xlsx          | 选取的香港地区的数据子集，用于研究地区效应                   |
| make and variant.xlsx   | 包含本项目所涉及的所有二手船变体                             |
| place and economic.xlsx | 为本项目所包含的地区及其经济情况的数据                       |
| region.xlsx             | 用于评估地区效应的数据子集                                   |

