setwd("C:/Users/Amir/Downloads")  

#Loaing the library

library(dplyr)
library(ggplot2)

# Reading the dataset 

mpg = read.csv("mpg.csv",
               stringsAsFactors = FALSE, 
               encoding = 'UTF-8')

View(mpg)
str(mpg)
nrow(mpg)
ncol(mpg)
colnames(mpg)

table(mpg$cty)
table(mpg$hwy)

summary(mpg$cty)
qplot(cty, data=mpg, geom="histogram", bins=40)

summary(mpg$hwy)
qplot(hwy, data=mpg, geom="histogram", bins=40)




# 1. plotting the relationship between cty and hwy

ggplot(mpg,
       aes(x=cty, y=hwy)) + 
  geom_point() + 
  geom_smooth(method="lm", formula=y ~ x) +
  xlab("city Average") +
  ylab("Highway Average") + 
  ggtitle("Distribution of mileage cty vs mty")

# i have fitted a regression line to see the relationship between cty and hwy, it is clear that average cty is lesser than average hwy.
# which proves the fact that on average car travel more distance on hwy and cty with same amount of fuel consumed.


# 2.
#A. 

ggplot(mpg, aes(cty, hwy)) + geom_point()

#Plot shows the relation between cty and hwy for each car in the datastet, the color is balck becasue there is no color specified and layer is point. 
# Datset is mpg and the used cols are cty for x-axis and hwy for y-axis.

#B. 

ggplot(diamonds, aes(carat, price)) + geom_point()

# Dataset used is diamond which comes by default with ggplot2 cran package, The graph shows the relationship between carat and price of diamond, it is 
# black by default as we have not specified any color, layer is point.

#c. 

ggplot(economics, aes(date, unemploy)) + geom_line()

#Dataset we are using is economic , again there is no color specified and the layer is line.
#we are plotting unemployment rate given time. 

#D. 

ggplot(mpg, aes(cty)) + geom_histogram()

# Have used mpg dataset, we have plotted a histogram for city average and there is non fill so the output is black. 









