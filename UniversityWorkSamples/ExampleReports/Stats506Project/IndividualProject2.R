#Author: David Li
#Purpose: Stats506 Code for Question #2: Analyzing School Size vs Scores for Individual Project
#Software Used: R and Dplyr package
#Data: data.gov for SAT 2010 and SAT 2012 scores, demographics file, ELA scores
#(https://catalog.data.gov/dataset/sat-college-board-2010...)
#(https://catalog.data.gov/dataset/sat-results-...)
#(http://schools.nyc.gov)


# Packages
library("dplyr", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
library("tidyr", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
library("readxl", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
library("ggplot2", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
library("gridExtra", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")


# Load / Clean Data

# SAT Math Data
SAT2010 = read.csv("~/Desktop/School/Stats506/Datasets/2010_SAT__College_Board__School_Level_Results.csv")
SAT2012 = read.csv("~/Desktop/School/Stats506/Datasets/2012_SAT_Results.csv")

columnnames1 = c("DBN", "SchoolName", "NumOfSATTakers2010", "Remove1", "MeanMathScore2010", "Remove2")
columnnames2 = c("DBN", "Remove3", "NumOfSATTakers2012", "Remove4", "MeanMathScore2012", "Remove5")
colnames(SAT2010) = columnnames1
colnames(SAT2012) = columnnames2

finalSAT = SAT2010 %>%
  inner_join(., SAT2012, by = "DBN") %>%
  transmute(DBN, SchoolName, NumOfSATTakers2010, NumOfSATTakers2012, MeanMathScore2010, MeanMathScore2012) %>%
  filter(NumOfSATTakers2010 != "s" & NumOfSATTakers2012 != "s" & MeanMathScore2010 != "s" & MeanMathScore2012 != "s")

# ELA Math Test Data
ELAMath = read_excel("~/Desktop/School/Stats506/Datasets/SchoolMathResults20062012Public.xlsx", range = "A7:P33468")

columnnames3 = c("DBN", "Grade", "Year", "Category", "NumberTested", "MeanScaleScore", "rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rm7", "rm8", "rm9", "rm10")
colnames(ELAMath) = columnnames3

finalELAMath = ELAMath %>%
  transmute(DBN, Grade, Year, NumberTested, MeanScaleScore) %>%
  filter(Grade == "All Grades") %>%
  filter(Year == 2012) %>%
  filter(MeanScaleScore != "s")

# Demographics Data
Demographic = read_excel("~/Desktop/School/Stats506/Datasets/DemographicSnapshot201213to201617Public_FINAL1.xlsx", sheet = "School")
columnnames4 = c("DBN", "SchoolName", "Year", "TotalEnroll", "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","aa","ab","ac","ad","ae","af","ag","ah","ai")
colnames(Demographic) = columnnames4

finalDemo = Demographic %>%
  transmute(DBN, SchoolName, Year, TotalEnroll) %>%
  filter(Year == "2012-13")

# ELA vs Enroll
ElaEnroll = finalELAMath %>%
  inner_join(., finalDemo, by = "DBN") 

# SAT vs Enroll
SATEnroll = finalSAT %>%
  inner_join(., finalDemo, by = "DBN")

# Generate graphs

plot1 = ggplot(ElaEnroll) + geom_point(aes(as.numeric(as.character(TotalEnroll)), as.numeric(as.character(MeanScaleScore))), color = "red") + geom_smooth(data = ElaEnroll, method = 'lm', formula = y ~ x, aes(as.numeric(as.character(TotalEnroll)), as.numeric(as.character(MeanScaleScore)))) + labs(title = "2012 New York ELA Math Scores", x = "Total Enrollment of the School in 2012", y = "Mean Scaled Scores")

plot2 = ggplot(SATEnroll) + geom_point(aes(as.numeric(as.character(TotalEnroll)), as.numeric(as.character(MeanMathScore2012)))) + geom_smooth(data = SATEnroll, method = 'lm', formula = y ~ x, aes(as.numeric(as.character(TotalEnroll)), as.numeric(as.character(MeanMathScore2012)))) + labs(title = "2012 New York SAT Math Scores", x = "Total Enrollment of the School in 2012", y = "Mean Scaled Scores")

plot1
plot2
