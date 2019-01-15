
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys


def SetPath(sc):
    global Path
    Path = "/mnt/data/backup/sdb2/usr/qd/"

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("Recommend").set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    print("master="+sc.master)
    return sc


def loadModel(sc):
    """load Model"""
    try:
        model = MatrixFactorizationModel.load(sc, Path+"ALSmodel")
        print("success...")
    except Exception:
        print("Failed!!")
    return model

def PrepareData(sc):
    """prepare data: movies.dat"""
    itemRDD = sc.textFile(Path+"ml-latest-small/movies.dat")
    movieTitle = itemRDD.map(lambda line: line.split("::")) \
        .map(lambda a: (int(a[0]), a[1])) \
        .collectAsMap()
    return movieTitle


def RecommendMovies(model,movieTitle,inputUserId):
    RecommendMovie = model.recommendProducts(inputUserId, int(input[1]))
    print("UserID : "+str(inputUserId)+"; Movies : "+input[1]+":")
    for p in RecommendMovie:
        print("Movie Name :"+str(movieTitle[p[1]]) + "Rating : " + str(p[2]))
    


def RecommendUsers(model,movieTitle,inputMovieId):
    RecommendUser = model.recommendUsers(inputMovieId, int(input[1]))
    print("MovieID : "+str(inputMovieId)+";"+input[1]+"users:")
    for p in RecommendUser:
        print("Movie Name :"+str(movieTitle[p[1]]) + "Rating : " + str(p[2]))

def RatingPredict(nodel, inputUserId, inputMovieId):
    rating = model.predict(inputUserId, inputMovieId)
    print("userID :" + str(inputUserId) + "; movieID :" + str(inputMovieId) + "Rating:" + str(rating))
		
def Recommend(model):
    if input[0][0] == "U":
        RecommendMovies(model, movieTitle, int(input[0][1:]))
    if input[0][0] == "M":
        RecommendUsers(model, movieTitle, int(input[0][1:]))
    if input[0][0] == "P":
        RatingPredict(model, int(input[0][1:]), int(input[1]))


if __name__ == "__main__":
    print("U/MuserID number")
    input = [i for i in sys.stdin.readline().strip().split(" ")]


    sc=CreateSparkContext()
    print("==========data prepare...==========")
    movieTitle = PrepareData(sc)
    print("==========loading Model...==========")
    model = loadModel(sc)
    print("==========Result==========")
    Recommend(model)
