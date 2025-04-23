# Describe and summarize the data. This is very useful as a preliminary step to capture basic data property. Distribution analysis, statistical exploration, correlation analysis, suitable transformation of variables and elimination of redundant variables, management of missing values.
# Produce at least two visualizations of the data that are meaningful to the report. Explain (in 1-2 sentence(s) for each figure) why do you think them are meaningful 

# what distribution do we want to see?
# 1. distribution of REPORTED_SATISFACTION

# what correlation do we want to see?
# between leave and reported satisfaction

# how can we handle repeated data?

# how can we handle missing values in data?
    # 1. count how many missing values we have per column

import csv
import os
import matplotlib.pyplot as plt

def plot_recorded_satisfaction():

    histogram = {
        "very_unsat": 0,
        "unsat": 0,
        "avg": 0,
        "sat": 0,
        "very_sat": 0
    }

    count = 0
    with open('./Customer_Churn.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # add code that adds one to the dictionary counter of that value in the col REPORTED_SATISFACTION
            value = row["REPORTED_SATISFACTION"]
            
            if value == "very_unsat":
                histogram["very_unsat"] += 1
            elif value == "unsat":
                histogram["unsat"] += 1
            elif value == "avg":
                histogram["avg"] += 1
            elif value == "sat":
                histogram["sat"] += 1
            elif value == "very_sat":
                histogram["very_sat"] += 1
            count += 1

    values_list = list(histogram.values())
    xs = list(range(len(values_list)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, values_list, marker='o', linestyle='-')
    plt.title("Reported Satisfaction Distribution")
    plt.xlabel("Reported Satisfaction")
    plt.ylabel("Number of customers")
    plt.xticks(xs, ['very_unsat', 'unsat', 'avg', 'sat', 'very_sat'])

    plt.show()

    fig.savefig("graph")
    print("count is", count)
        
def correlation_leave_and_recorded_satisfaction():

    histogram = {
        "very_unsat_stay": 0,
        "very_unsat_leave": 0,
        "unsat_stay": 0,
        "unsat_leave": 0,
        "avg_stay": 0,
        "avg_leave": 0,
        "sat_stay": 0,
        "sat_leave": 0,
        "very_sat_stay": 0,
        "very_sat_leave": 0
    }

    with open('./Customer_Churn.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # add code that adds one to the dictionary counter of that value in the col REPORTED_SATISFACTION
            value = row["REPORTED_SATISFACTION"]
            value2 = row["LEAVE"]
            if value == "very_unsat":
                if value2 == "STAY":
                    histogram["very_unsat_stay"] +=1
                else:
                    histogram["very_unsat_leave"] +=1

            elif value == "unsat":
                if value2 == "STAY":
                    histogram["unsat_stay"] +=1
                else:
                    histogram["unsat_leave"] +=1
            elif value == "avg":
                if value2 == "STAY":
                    histogram["avg_stay"] +=1
                else:
                    histogram["avg_leave"] +=1
            elif value == "sat":
                if value2 == "STAY":
                    histogram["sat_stay"] +=1
                else:
                    histogram["sat_leave"] +=1
            elif value == "very_sat":
                if value2 == "STAY":
                    histogram["very_sat_stay"] +=1
                else:
                    histogram["very_sat_leave"] +=1
    
    values_list = list(histogram.values())
    xs = list(range(len(values_list)))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(xs, values_list)
    plt.show()
    plt.title("Leave and Reported Satisfaction")
    plt.xlabel("Leave and Reported Satisfaction")
    plt.ylabel("Number of customers")
    #plt.yticks(reported_satisfaction, ["very_unsat", "unsat", "avg", "sat", "very_sat"])
    #plt.xticks(values_list, ["very_unsat_stay", "very_unsat_leave", "unsat_stay", "unsat_leave", "avg_stay", "avg_leave", "sat_stay", "sat_leave", "very_sat_stay", "very_sat_leave"])
    plt.xticks(xs, ["very_unsat_stay", "very_unsat_leave", "unsat_stay", "unsat_leave", "avg_stay", "avg_leave", "sat_stay", "sat_leave", "very_sat_stay", "very_sat_leave"])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.subplots_adjust(bottom=0.18)


    plt.show()

    fig.savefig("graph3")


def avg_for_stay_and_leave():
    
    #output: printed numbers for each avg
    return 0

def avg_for_sat_and_unsat():
    #group sat/very_sat together

    #group unsat/very_unsat together

    #output: printed numbers for each avg
    return 0

def correlation_leave_and_considering_change_of_plan():

    # output: bar chart
    return 0

if __name__ == "__main__":
    # plot_recorded_satisfaction() 
    correlation_leave_and_recorded_satisfaction()

# all attributes
# COLLEGE	INCOME	OVERAGE	LEFTOVER	HOUSE	HANDSET_PRICE	OVER_15MINS_CALLS_PER_MONTH	AVERAGE_CALL_DURATION	REPORTED_SATISFACTION	REPORTED_USAGE_LEVEL	CONSIDERING_CHANGE_OF_PLAN      LEAVE


# extra notes 
"""
Difference between satisfied and unsatisfied customers:
- calculate averages or modes and compare for each attribute 

 Difference between leave and stay customers:
- calculate averages or modes and compare for each attribute 

Correlation between considering change of plan and leave/stay
If most ppl were not considering, could signify sudden leave (maybe competitors price was too good, or they had a limited promotion that made customers switch quickly) 


"""