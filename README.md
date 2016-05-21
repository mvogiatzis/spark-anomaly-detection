Using Spark for Anomaly (Fraud) Detection
=========================================

The project identifies anomalies in datasets using an algorithm explained [on this post]().

The two fraud-detection datasets provided in the test resources were created using fake data.

Running on Spark
----------------

First compile:

```
sbt package
```

Run the tests:

```
sbt test
```

Download Spark (1.6.1) and put it in your Path from here: http://spark.apache.org/downloads.html

Run the project locally:

```
spark-submit --class "MainRun" --master local[8] target/scala-2.10/anomaly-detection_2.10-1.0.jar
```

