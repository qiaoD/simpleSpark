
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS

def SetPath(sc):
    global Path
    Path = "/mnt/data/backup/sdb2/usr/qd/"


def CreateSparkContext():
    sparkConf = SparkConf().setAppName("Recommend").set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    print("master="+sc.master)
    return sc

def PrepareData(sc):
    """Data Prepare ratings.dat -> RDD Data"""
    rawUserData = sc.textFile(Path + "ml-latest-small/ratings.dat")
    rawRatings = rawUserData.map(lambda line: line.split("::")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD


def SaveModel(sc):
    """Save Model"""
    try:
        model.save(sc, Path+"ALSmodel")
        print("Model saved...")
    except Exception:
        print("Failed!!!")


if __name__ == "__main__":
    sc = CreateSparkContext()
    print("==========Prepare Data...==========")
    ratingsRDD = PrepareData(sc)
    print("========== Training... ============")
    model = ALS.train(ratingsRDD, 5, 10, 0.1)
    print("========== Saving Model. ==========")
    SaveModel(sc)
