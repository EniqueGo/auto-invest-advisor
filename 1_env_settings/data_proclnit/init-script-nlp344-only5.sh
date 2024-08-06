#!/bin/bash
set -euxo pipefail

#Apache Spark 3.3.x, 3.2.x, 3.1.x, 3.0.x
#2.2 (Debian 11 Hadoop 3.3, Spark 3.3)


apt-get -y update
apt-get install openjdk-11-jdk-headless -y
apt install python3
apt install python3-pip -y

# Install Spark NLP
pip install spark-nlp==4.2.0
pip install keras==2.8
pip install transformers
pip install torch
pip install ktrain
pip install tensorflow==2.10
pip install numpy==1.25.2
pip install py4j -y

# Add properties= spark.jars = gs://stonkgo2-spark-bucket/init-script/spark-nlp_2.12-4.2.0.jar