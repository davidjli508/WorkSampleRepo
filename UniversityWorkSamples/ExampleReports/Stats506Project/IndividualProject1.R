#Author: David Li
#Purpose: Stats506 Code for Question #1: 2010 vs 2012 SAT scores for Individual Project
#Software Used: R and Dplyr package
#Data: data.gov for SAT 2010 and SAT 2012 scores
#(https://catalog.data.gov/dataset/sat-college-board-2010...)
#(https://catalog.data.gov/dataset/sat-results-...)


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
  filter(NumOfSATTakers2010 != "s" & NumOfSATTakers2012 != "s" & MeanMathScore2010 != "s" & MeanMathScore2012 != "s") %>%
  mutate(difference = as.numeric(MeanMathScore2012) - as.numeric(MeanMathScore2010))

seq1 = seq(from = 1, to = dim(finalSAT)[1])

finalSAT = finalSAT %>%
  mutate(Index = seq1)


# ELA Math Test Data
ELAMath = read_excel("~/Desktop/School/Stats506/Datasets/SchoolMathResults20062012Public.xlsx", range = "A7:P33468")

columnnames3 = c("DBN", "Grade", "Year", "Category", "NumberTested", "MeanScaleScore", "rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rm7", "rm8", "rm9", "rm10")
colnames(ELAMath) = columnnames3

finalELAMath = ELAMath %>%
  transmute(DBN, Grade, Year, NumberTested, MeanScaleScore) %>%
  filter(Grade == "All Grades") %>%
  filter(Year == 2012) %>%
  filter(MeanScaleScore != "s")

seq2 = seq(from = 1, to = dim(finalELAMath)[1])

finalELAMath = finalELAMath %>%
  mutate(Index = seq2)

# Demographics Data
Demographic = read_excel("~/Desktop/School/Stats506/Datasets/DemographicSnapshot201213to201617Public_FINAL1.xlsx", sheet = "School")
columnnames4 = c("DBN", "SchoolName", "Year", "TotalEnroll", "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","aa","ab","ac","ad","ae","af","ag","ah","ai")
colnames(Demographic) = columnnames4

finalDemo = Demographic %>%
  transmute(DBN, SchoolName, Year, TotalEnroll)

# Generate graphs

plot1 = ggplot(finalSAT) + geom_point(aes(as.numeric(as.character(Index)), as.numeric(as.character(MeanMathScore2010))), color = "deepskyblue1") + geom_point(aes(as.numeric(as.character(Index)), as.numeric(as.character(MeanMathScore2012))), color = "darkorange1") + labs(title = "New York: 2010(Blue) and 2012(Orange)", x = "Unique Index Assignment per School", y = "SAT Mean Math Score")

plot2 = ggplot(finalSAT) + geom_point(aes(as.numeric(as.character(Index)), as.numeric(as.character(difference)))) + labs(title = "NY Score Differences (2012-2010)", x = "Unique Index Assignment per School", y = "Difference Mean Math SAT Score") + geom_hline(yintercept = 0)

plot1
plot2
