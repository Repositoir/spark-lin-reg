# Survivability using Linear Regression

This model predicts survivability of a passenger on the Titanic using 4 categories of 
`Sex`, `Embarked`, `Age` and `Fare`. This was done for the Kaggle Titanic - Machine Learning from Disaster
competition.

> Note: Since this model still requires a few changes, it has not been made an official submission for the competition yet.

## Requirements

- Java Development Kit version 11 or lower.
- Spark version 3.x
- Scala version 2.13.x
- Scala Build Tool (sbt)

All build configurations are given in the `build.sbt` file. Only the adequate JDK version should be manually selected.

To avoid Java Exceptions from `sun.nio.ch`, ensure to add `--add-exports java.base/sun.nio.ch=ALL-UNNAMED` into
your JVM options.

## The Model

In the `Main.scala` file and subsequently the `main` method, the following steps are undertaken

- Data Cleaning by removing `NULL` values using Spark's `sql.na.fill`
and filling them with appropriate values
- Indexing some binary data columns such as `Embarked` and `Sex`
- Created a `Pipeline` to fit and transform the above data
- Create a `LinearRegression()` model using `spark-mllib` and make prediction on survivability using the specified columns. 