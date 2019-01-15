# -*-coding: utf-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS

def SetPath(sc):
    """定义全局变量Path，配置文件读取"""
    global Path
    if sc.master[0:5] == "local":   # 当前为本地路径，读取本地文件
        Path = "file:/home/yyf/pythonwork/PythonProject/"
    else:       # 不是本地路径，有可能是YARN Client或Spark Stand Alone必须读取HDFS文件
        Path = "hdfs://master:9000/user/yyf/"


def CreateSparkContext():
    """定义CreateSparkContext函数便于创建SparkContext实例"""
    sparkConf = SparkConf() \
             .setAppName("Recommend") \
             .set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    print("master="+sc.master)
    return sc

def PrepareData(sc):
    """数据预处理:读取u.data文件，转化为ratingsRDD数据类型"""
    rawUserData = sc.textFile(Path + "data/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD


def SaveModel(sc):
    """存储模型"""
    try:
        model.save(sc, Path+"ALSmodel")
        print("模型已存储")
    except Exception:
        print("模型已存在,先删除后创建")


if __name__ == "__main__":
    sc = CreateSparkContext()
    print("==========数据准备阶段==========")
    ratingsRDD = PrepareData(sc)
    print("========== 训练阶段 ============")
    print(" 开始ALS训练，参数rank=5,iterations=10,lambda=0.1")
    model = ALS.train(ratingsRDD, 5, 10, 0.1)
    print("========== 存储model ==========")
    SaveModel(sc)
