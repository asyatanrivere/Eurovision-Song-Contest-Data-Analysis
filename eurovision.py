import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os

data_path="eurovision2026.csv"
output="images"
os.makedirs(output,exist_ok=True)

# LOAD DATA
#-----------------------
def load_data(path):
    df =pd.read_csv(path)
    return df

# INSPECT DATA
#-----------------------
def inspect_data(df):
    print("\n--HEAD--")
    print(df.head(10))

    print("\n--TAIL--")
    print(df.tail(10))

    print("\n--DESCRIBE--")
    print(df.describe())

    print("\n--INFO--")
    print(df.info())

    print("\n--COLUMNS--")
    print(df.columns)

    print("\n--CORRELATION--")
    print(df.corr(numeric_only=True))

    print("\n--DUPLICATED--")
    print(df.duplicated().sum())

    print("\n--IS NULL--")
    print(df.isnull().sum())

# CLEANING DATA
#-----------------------
def cleaning_data(df):
    df=df.copy()
    df.drop(columns=["artist_url","running_order","image_url","event_url","country_emoji","rank_ordinal","qualified"],inplace=True)
    df.dropna(subset=["host_city","host_country","artist","song","artist_country","total_points","rank","winner"],inplace=True)
    df["total_points"]=df["total_points"].astype(int)
    df["rank"]=df["rank"].astype(int)
    return df

# YEARLY NUMBER OF PARTICIPANTS ANALYSIS
#----------------------- 
def yearly_number_of_participants(df):
    yearly_values_count=df["year"].value_counts().sort_index(ascending=True)
    plt.figure(figsize=(15,5))
    sb.lineplot(x=yearly_values_count.index,y=yearly_values_count.values).set_xticks(df['year'].unique())
    plt.title("Number of participating countries by year")
    plt.xlabel("Years")
    plt.ylabel("Number of participating countries")
    plt.xticks(rotation=45,ha="right")
    plt.margins(x=0)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/yearly_values_count.png")
    plt.show()

# HOSTING COUNTRIES ANALYSIS
#-----------------------
def hosting_countries(df):
    hostcity_number=df["host_country"].value_counts()
    plt.figure(figsize=(8,8))
    sb.barplot(y=hostcity_number.index,x=hostcity_number.values)
    plt.title("Hosting Countries and Number of Hosting")
    plt.xlabel("Number of Hosting")
    plt.ylabel("Hosting Countries")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/hosting_countries.png")
    plt.show()

# TOP 30 MOST PARTICIPATING ARTISTS ANALYSIS
#-----------------------
def top_30_most_participating_artists(df):
    top_30_artist=df["artist"].value_counts().head(30)
    plt.figure(figsize=(7,7))
    sb.barplot(y=top_30_artist.index,x=top_30_artist.values)
    plt.title("Top 30 Most Participating Artists")
    plt.xlabel("Number of Participants")
    plt.ylabel("Artist")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/top_30_most_participating_artists.png")
    plt.show()

# CORRELATION HEATMAP
#-----------------------
def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8,8))
    sb.heatmap(data=corr, annot=True, cmap='RdYlBu', linewidths=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45, ha="right")
    plt.title("Correlation ")
    plt.savefig(f"{output}/correlation_heatmap.png")
    plt.show()

# TOP 20 HIGHEST RATED SONGS ANALYSIS
#-----------------------
def top_20_highest_rated_songs(df):
    plt.figure(figsize=(15,5))
    top20songs = df.sort_values(by="total_points", ascending=False).head(20)
    sb.barplot(data=top20songs,y="song",x="total_points")
    plt.title("The Top 20 Highest-Rated Songs")
    plt.xlabel("Total Points")
    plt.ylabel("Songs")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(f"{output}/top_20_highest-rated_songs.png")
    plt.show()

# ANALYSIS OF COUNTRIES THAT HAVE MOST WINS 
#-----------------------
def countries_with_most_eurovision_wins_analysis(df):
    plt.figure(figsize=(7,6))

    winners=df[df["winner"]==True]
    winner_countries=winners["artist_country"].value_counts()
    sb.barplot(y=winner_countries.index,x=winner_countries.values)

    plt.title("Countries with the Most Eurovision Wins")
    plt.xlabel(" Countries")
    plt.ylabel("Total Number of Wins")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(f"{output}/countries_with_most_eurovision_wins.png")
    plt.show()

# ANALYSIS OF COUNTRIES WITH SUM OF TOTAL POINTS (ALL YEARS)
#-----------------------
def countries_with_sum_of_total_points_of_all_years_analysis(df):
    plt.figure(figsize=(8,8))
    most_ranked_countries=df.groupby("artist_country")["total_points"].sum().sort_values(ascending=False)
    sb.barplot(y=most_ranked_countries.index,x=most_ranked_countries.values)
    plt.title("Countries with the Sum of Total Points of All Years")
    plt.ylabel("Countries")
    plt.xlabel("Sum of All Total Points")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(f"{output}/countries_with_sum_of_total_points_of_all_years.png")
    plt.show()

# ANALYSIS OF COUNTRIES WITH ALL RANKS 
#-----------------------
def countries_with_all_ranks_analysis(df):
    plt.figure(figsize=(8,10))
    sb.scatterplot(data=df,x="rank",y="artist_country")
    plt.title("Countries with All the Rankings They Achieved")
    plt.ylabel("Countries")
    plt.xlabel("Ranks")
    plt.tight_layout()
    plt.savefig(f"{output}/countries_with_all_ranks_analysis.png")
    plt.show()

# ANALYSIS OF RANKS VS TOTAL POINTS
#-----------------------
def analysis_ranks_vs_total_points(df):
    sb.scatterplot(data=df,x="rank",y="total_points",size=0.1)
    plt.title("Ranks vs Total Points")
    plt.ylabel("Total Points")
    plt.xlabel("Ranks")
    plt.tight_layout()
    plt.savefig(f"{output}/ranks_vs_total_points.png")
    plt.show()

# MAIN PIPELINE
#-----------------------
def main():
    df =pd.read_csv(data_path)
    inspect_data(df)
    df= cleaning_data(df)
    yearly_number_of_participants(df)
    hosting_countries(df)
    top_30_most_participating_artists(df)
    correlation_analysis(df)
    top_20_highest_rated_songs(df)
    countries_with_most_eurovision_wins_analysis(df)
    countries_with_all_ranks_analysis(df)
    analysis_ranks_vs_total_points(df)

# RUN
#-----------------------
if __name__=="__main__":
    main()