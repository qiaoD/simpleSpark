# -*-coding: utf-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys


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


def loadModel(sc):
    """载入训练好的推荐模型"""
    try:
        model = MatrixFactorizationModel.load(sc, Path+"ALSmodel")
        print("载入模型成功")
    except Exception:
        print("模型不存在, 请先训练模型")
    return model

def PrepareData(sc):
    """数据准备：准备u.item文件返回电影id-电影名字典（对照表）"""
    itemRDD = sc.textFile(Path+"data/u.item")
    movieTitle = itemRDD.map(lambda line: line.split("|")) \
        .map(lambda a: (int(a[0]), a[1])) \
        .collectAsMap()
    return movieTitle


def RecommendMovies(model,movieTitle,inputUserId):
    RecommendMovie = model.recommendProducts(inputUserId, int(input[1]))
    print("对用户ID为"+str(inputUserId)+"的用户推荐下列"+input[1]+"部电影：")
    for p in RecommendMovie:
        print("对编号为" + str(p[0]) + "的用户" + "推荐电影" + str(movieTitle[p[1]]) + "推荐评分为" + str(p[2]))


def RecommendUsers(model,movieTitle,inputMovieId):
    RecommendUser = model.recommendUsers(inputMovieId, int(input[1]))
    print("对电影ID为"+str(inputMovieId)+"的电影推荐给下列"+input[1]+"个用户：")
    for p in RecommendUser:
        print("对编号为" + str(p[0]) + "的用户" + "推荐电影" + str(movieTitle[p[1]]) + "推荐评分为" + str(p[2]))


def Recommend(model):
    """根据参数进行推荐"""
    if input[0][0] == "U":
        RecommendMovies(model, movieTitle, int(input[0][1:]))
    if input[0][0] == "M":
        RecommendUsers(model, movieTitle, int(input[0][1:]))


if __name__ == "__main__":
    print("请输入2个参数, 第一个参数指定推荐模式（用户/电影）, 第二个参数为推荐的数量如U666 10表示向用户666推荐10部电影")
    input = [i for i in sys.stdin.readline().strip().split(" ")]


    sc=CreateSparkContext()
    print("==========数据准备==========")
    movieTitle = PrepareData(sc)
    print("==========载入模型==========")
    model = loadModel(sc)
    print("==========进行推荐==========")
    Recommend(model)
