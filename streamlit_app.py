import pandas as pd
import streamlit as st
#from venny4py.venny4py import *

df_2024 = pd.read_excel('Spotify Top 100 2024.xlsx', sheet_name='2024 Full View').iloc[:, :-4]
df_2023 = pd.read_excel('Spotify Top 100 2024.xlsx', sheet_name='2023 Full View').iloc[:, :-4]
df_2024['Release Date'] = df_2024['Release Date'].astype(str)

overlap_24 = df_2024.iloc[:,0:5].dropna(thresh=3)
overlap_24 = overlap_24.merge(df_2024.loc[:, [df_2024.columns[0]] + list(df_2024.columns[5:])], on='track_id')
overlap_24 = overlap_24[['Song', 'Artist', 'Zach', 'Bryce', 'Maggie', 'Jamie', 'Release Date']]
overlap_24['Total Listeners'] = 0
for rec in overlap_24.index:
    if not pd.isna(overlap_24.loc[rec, 'Zach']):
        overlap_24.loc[rec, 'Total Listeners'] += 1
    if not pd.isna(overlap_24.loc[rec, 'Maggie']):
        overlap_24.loc[rec, 'Total Listeners'] += 1
    if not pd.isna(overlap_24.loc[rec, 'Jamie']):
        overlap_24.loc[rec, 'Total Listeners'] += 1
    if not pd.isna(overlap_24.loc[rec, 'Bryce']):
        overlap_24.loc[rec, 'Total Listeners'] += 1
overlap_24 = overlap_24.sort_values(by='Total Listeners', ascending=False).reset_index(drop=True)
# overlap_24['Zach'] = overlap_24['Zach'].fillna("")
# overlap_24['Maggie'] = overlap_24['Maggie'].fillna("")
# overlap_24['Jamie'] = overlap_24['Jamie'].fillna("")
# overlap_24['Bryce'] = overlap_24['Bryce'].fillna("")

zach_overlaps = df_2024[df_2024['Zach'].notna()]
temp = df_2023[df_2023['Zach'].notna()][['track_id', 'Zach']]
temp = temp.rename(columns={'Zach': 'Zach_2023'})
zach_overlaps = zach_overlaps.merge(temp, on='track_id', how='left')
zach_overlaps = zach_overlaps[zach_overlaps['Zach_2023'].notna()]
zach_overlaps = zach_overlaps[['Song','Artist', 'Zach', 'Zach_2023']]
zach_overlaps['YoY Change'] = zach_overlaps['Zach_2023'] - zach_overlaps['Zach']
zach_overlaps = zach_overlaps.reset_index(drop=True)

maggie_overlaps = df_2024[df_2024['Maggie'].notna()]
temp = df_2023[df_2023['Maggie'].notna()][['track_id', 'Maggie']]
temp = temp.rename(columns={'Maggie': 'Maggie_2023'})
maggie_overlaps = maggie_overlaps.merge(temp, on='track_id', how='left')
maggie_overlaps = maggie_overlaps[maggie_overlaps['Maggie_2023'].notna()]
maggie_overlaps = maggie_overlaps[['Song','Artist', 'Maggie', 'Maggie_2023']]
maggie_overlaps['YoY Change'] = maggie_overlaps['Maggie_2023'] - maggie_overlaps['Maggie']
maggie_overlaps = maggie_overlaps.reset_index(drop=True)

jamie_overlaps = df_2024[df_2024['Jamie'].notna()]
temp = df_2023[df_2023['Jamie'].notna()][['track_id', 'Jamie']]
temp = temp.rename(columns={'Jamie': 'Jamie_2023'})
jamie_overlaps = jamie_overlaps.merge(temp, on='track_id', how='left')
jamie_overlaps = jamie_overlaps[jamie_overlaps['Jamie_2023'].notna()]
jamie_overlaps = jamie_overlaps[['Song','Artist', 'Jamie', 'Jamie_2023']]
jamie_overlaps['YoY Change'] = jamie_overlaps['Jamie_2023'] - jamie_overlaps['Jamie']
jamie_overlaps = jamie_overlaps.reset_index(drop=True)

bryce_overlaps = df_2024[df_2024['Bryce'].notna()]
temp = df_2023[df_2023['Bryce'].notna()][['track_id', 'Bryce']]
temp = temp.rename(columns={'Bryce': 'Bryce_2023'})
bryce_overlaps = bryce_overlaps.merge(temp, on='track_id', how='left')
bryce_overlaps = bryce_overlaps[bryce_overlaps['Bryce_2023'].notna()]
bryce_overlaps = bryce_overlaps[['Song','Artist', 'Bryce', 'Bryce_2023']]
bryce_overlaps['YoY Change'] = bryce_overlaps['Bryce_2023'] - bryce_overlaps['Bryce']
bryce_overlaps = bryce_overlaps.reset_index(drop=True)

influence = df_2024.copy()
temp = df_2023.copy()
# rename Zach to Zach_2023, Maggie to Maggie_2023, Jamie to Jamie_2023, Bryce to Bryce_2023
temp = temp.rename(columns={'Zach': 'Zach_2023', 'Maggie': 'Maggie_2023', 'Jamie': 'Jamie_2023', 'Bryce': 'Bryce_2023'})
temp = temp[['track_id', 'Zach_2023', 'Maggie_2023', 'Jamie_2023', 'Bryce_2023']]
influence = influence.merge(temp, on='track_id', how='left')
influence = influence[influence['Zach_2023'].notna() | influence['Maggie_2023'].notna() | influence['Jamie_2023'].notna() | influence['Bryce_2023'].notna()]
influence['influence'] = 0
for rec in influence.index:
    if not pd.isna(influence.loc[rec, 'Zach']) and pd.isna(influence.loc[rec, 'Zach_2023']):
        influence.loc[rec, 'influence'] = 1
    if not pd.isna(influence.loc[rec, 'Maggie']) and pd.isna(influence.loc[rec, 'Maggie_2023']):
        influence.loc[rec, 'influence'] = 1
    if not pd.isna(influence.loc[rec, 'Jamie']) and pd.isna(influence.loc[rec, 'Jamie_2023']):
        influence.loc[rec, 'influence'] = 1
    if not pd.isna(influence.loc[rec, 'Bryce']) and pd.isna(influence.loc[rec, 'Bryce_2023']):
        influence.loc[rec, 'influence'] = 1
influence = influence[influence['influence'] == 1].reset_index(drop=True)
influence['2024 Listener(s)'] = ""
influence['2023 Listener(s)'] = ""
for rec in influence.index:
    if not pd.isna(influence.loc[rec, 'Zach']):
        influence.loc[rec, '2024 Listener(s)'] += "Zach "
    if not pd.isna(influence.loc[rec, 'Maggie']):
        influence.loc[rec, '2024 Listener(s)'] += "Maggie "
    if not pd.isna(influence.loc[rec, 'Jamie']):
        influence.loc[rec, '2024 Listener(s)'] += "Jamie "
    if not pd.isna(influence.loc[rec, 'Bryce']):
        influence.loc[rec, '2024 Listener(s)'] += "Bryce "
    if not pd.isna(influence.loc[rec, 'Zach_2023']):
        influence.loc[rec, '2023 Listener(s)'] += "Zach "
    if not pd.isna(influence.loc[rec, 'Maggie_2023']):
        influence.loc[rec, '2023 Listener(s)'] += "Maggie "
    if not pd.isna(influence.loc[rec, 'Jamie_2023']):
        influence.loc[rec, '2023 Listener(s)'] += "Jamie "
    if not pd.isna(influence.loc[rec, 'Bryce_2023']):
        influence.loc[rec, '2023 Listener(s)'] += "Bryce "
influence = influence[['Song', 'Artist', '2023 Listener(s)', '2024 Listener(s)']]
# replace space in 2024 Listener(s) and 2023 Listener(s) with comma
influence['2024 Listener(s)'] = influence['2024 Listener(s)'].str.replace(" ", ", ")
influence['2023 Listener(s)'] = influence['2023 Listener(s)'].str.replace(" ", ", ")
# remove trailing comma
influence['2024 Listener(s)'] = influence['2024 Listener(s)'].str.rstrip(", ")
influence['2023 Listener(s)'] = influence['2023 Listener(s)'].str.rstrip(", ")

# take df_2024 and create a dataframe with columns Artist, Zach, Maggie, Jamie, Bryce, and Total where Total is the sum of Zach, Maggie, Jamie, and Bryce
top_artists = pd.DataFrame(columns=['Artist', 'Zach', 'Maggie', 'Jamie', 'Bryce', 'Total'])
#artists = df_2024['Artist'].unique()+df_2024['Featured Artist'].unique()
artists = df_2024['Artist'].unique().tolist()
artists.extend(df_2024['Featured Artist'].unique().tolist())
artists = list(set(artists))
for artist in artists:
    temp = df_2024[(df_2024['Artist'] == artist) | (df_2024['Featured Artist'] == artist)]
    temp = temp[['Zach', 'Maggie', 'Jamie', 'Bryce']]
    temp = temp.count()
    temp = pd.DataFrame(temp).T
    temp['Artist'] = artist
    temp['Total'] = temp['Zach'] + temp['Maggie'] + temp['Jamie'] + temp['Bryce']
    top_artists = pd.concat([top_artists, temp], ignore_index=True)
top_artists = top_artists[['Artist', 'Zach', 'Maggie', 'Jamie', 'Bryce', 'Total']]
top_artists = top_artists.sort_values(by='Total', ascending=False).reset_index(drop=True).iloc[:-1,:]
top_artists['Total Unique Songs'] = 0
for artist in top_artists['Artist']:
    temp = df_2024[df_2024['Artist'] == artist]
    top_artists.loc[top_artists['Artist'] == artist, 'Total Unique Songs'] = len(temp)

# Ensure a fresh copy of the DataFrame to avoid caching issues in Streamlit
top_artists = top_artists.copy()

# Define listeners
listeners = ['Zach', 'Maggie', 'Jamie', 'Bryce']

# Create an empty list to store rows
rows = []

# Loop through each listener and find artists they listened to exclusively
for listener in listeners:
    other_listeners = [l for l in listeners if l != listener]
    
    # Condition: The listener has played it, and no one else has
    condition = (top_artists[listener] > 0) & (top_artists[other_listeners].sum(axis=1) == 0)
    
    # Append results to list
    for _, row in top_artists.loc[condition, ['Artist', listener]].iterrows():
        rows.append({'Artist': row['Artist'], 'Listener': listener, 'Total': row[listener]})

# Convert list to DataFrame
top_artists_only_one_listener = pd.DataFrame(rows)

# Sort by Total plays
top_artists_only_one_listener = top_artists_only_one_listener.sort_values(by='Total', ascending=False).reset_index(drop=True)

# Ensure a fresh copy of the DataFrame to avoid caching issues in Streamlit
top_artists = top_artists.copy()

# Define listeners
listeners = ['Zach', 'Maggie', 'Jamie', 'Bryce']

# Create an empty list to store rows
rows = []

# Iterate through each listener and extract non-zero values
for listener in listeners:
    filtered_df = top_artists[top_artists[listener] > 0][['Artist', listener]]
    
    # Convert to the desired format
    for _, row in filtered_df.iterrows():
        rows.append({'Artist': row['Artist'], 'Listener': listener, 'Total': row[listener]})

# Convert list to DataFrame
top_artists_by_listener = pd.DataFrame(rows)

# Sort by Total plays
top_artists_by_listener = top_artists_by_listener.sort_values(by='Total', ascending=False).reset_index(drop=True)


unique_artists_per_listener = pd.DataFrame(columns=['Listener', 'Unique Artists'])
for listener in ['Zach', 'Maggie', 'Jamie', 'Bryce']:
    temp = len(top_artists[top_artists[listener] > 0])
    temp = pd.DataFrame({'Listener': [listener], 'Unique Artists': [temp]})
    unique_artists_per_listener = pd.concat([unique_artists_per_listener, temp], ignore_index=True)
unique_artists_per_listener = unique_artists_per_listener.sort_values(by='Unique Artists', ascending=False).reset_index(drop=True)

temp = df_2024.copy()
temp['decade'] = ""
temp['decade'] = temp['Release Date'].str[:3]
temp['decade'] = temp['decade'].astype(int)
temp['decade'] = temp['decade'] * 10
temp['decade'] = temp['decade'].astype(str)
temp['decade'] = temp['decade'] + "s"
decades = temp.groupby('decade').count()
decades = decades[['Zach', 'Maggie', 'Jamie', 'Bryce']]
decades = decades.reset_index()

song_count_2024 = df_2024.copy()
song_count_2024 = song_count_2024[['Zach', 'Maggie', 'Jamie', 'Bryce', 'Release Date']]
# get release date year = 2024
song_count_2024['Release Date'] = song_count_2024['Release Date'].str[:4]
song_count_2024 = song_count_2024[song_count_2024['Release Date'] == '2024']
song_count_2024 = song_count_2024.count()
song_count_2024 = pd.DataFrame(song_count_2024).T
# remove Release Date column
song_count_2024 = song_count_2024.drop(columns='Release Date')
# add column add front Titled Year with 2024
song_count_2024.insert(0, 'Year', '2024     ')


most_popular = df_2024.sort_values(by='Popularity', ascending=False).reset_index(drop=True).head(10)
most_popular['Listener(s)'] = ""
for rec in most_popular.index:
    if not pd.isna(most_popular.loc[rec, 'Zach']):
        most_popular.loc[rec, 'Listener(s)'] += "Zach "
    if not pd.isna(most_popular.loc[rec, 'Maggie']):
        most_popular.loc[rec, 'Listener(s)'] += "Maggie "
    if not pd.isna(most_popular.loc[rec, 'Jamie']):
        most_popular.loc[rec, 'Listener(s)'] += "Jamie "
    if not pd.isna(most_popular.loc[rec, 'Bryce']):
        most_popular.loc[rec, 'Listener(s)'] += "Bryce "
most_popular = most_popular[['Song', 'Artist', 'Listener(s)', 'Popularity']]
# replace space in 2024 Listener(s) and 2023 Listener(s) with comma
most_popular['Listener(s)'] = most_popular['Listener(s)'].str.replace(" ", ", ")
# remove trailing comma
most_popular['Listener(s)'] = most_popular['Listener(s)'].str.rstrip(", ")

least_popular = df_2024.sort_values(by='Popularity', ascending=True).reset_index(drop=True).head(10)
least_popular['Listener(s)'] = ""
for rec in least_popular.index:
    if not pd.isna(least_popular.loc[rec, 'Zach']):
        least_popular.loc[rec, 'Listener(s)'] += "Zach "
    if not pd.isna(least_popular.loc[rec, 'Maggie']):
        least_popular.loc[rec, 'Listener(s)'] += "Maggie "
    if not pd.isna(least_popular.loc[rec, 'Jamie']):
        least_popular.loc[rec, 'Listener(s)'] += "Jamie "
    if not pd.isna(least_popular.loc[rec, 'Bryce']):
        least_popular.loc[rec, 'Listener(s)'] += "Bryce "
least_popular = least_popular[['Song', 'Artist', 'Listener(s)', 'Popularity']]
# replace space in 2024 Listener(s) and 2023 Listener(s) with comma
least_popular['Listener(s)'] = least_popular['Listener(s)'].str.replace(" ", ", ")
# remove trailing comma
least_popular['Listener(s)'] = least_popular['Listener(s)'].str.rstrip(", ")

avg_popularity = pd.DataFrame(columns=['Listener', 'Avg Popularity'])
for listener in ['Zach', 'Maggie', 'Jamie', 'Bryce']:
    temp = df_2024[df_2024[listener].notna()]
    temp = temp['Popularity'].mean()
    temp = pd.DataFrame({'Listener': [listener], 'Avg Popularity': [temp]})
    avg_popularity = pd.concat([avg_popularity, temp], ignore_index=True)
avg_popularity = avg_popularity.sort_values(by='Avg Popularity', ascending=False).reset_index(drop=True)

import numpy as np
top_albums = pd.DataFrame(columns=['Album', 'Artist', 'Zach', 'Maggie', 'Jamie', 'Bryce', 'Total'])
albums = df_2024['Album'].unique()
for album in albums:
    try:
        temp = df_2024[df_2024['Album'] == album]
        temp = temp[['Zach', 'Maggie', 'Jamie', 'Bryce']]
        temp = temp.count()
        temp = pd.DataFrame(temp).T
        temp['Album'] = album
        temp['Artist'] = df_2024[df_2024['Album'] == album]['Artist'].unique()[0]
        temp['Total'] = temp['Zach'] + temp['Maggie'] + temp['Jamie'] + temp['Bryce']
        top_albums = pd.concat([top_albums, temp], ignore_index=True)
    except:
        continue
top_albums = top_albums[['Album', 'Artist', 'Zach', 'Maggie', 'Jamie', 'Bryce', 'Total']]
top_albums = top_albums.sort_values(by='Total', ascending=False).reset_index(drop=True).iloc[:-1,:]
# count the rows per album
top_albums['Total Unique Songs'] = 0
for album in top_albums['Album']:
    temp = df_2024[df_2024['Album'] == album]
    top_albums.loc[top_albums['Album'] == album, 'Total Unique Songs'] = len(temp)
    
# bryce = df_2024[df_2024['Bryce'].notna()]['track_id']
# zach = df_2024[df_2024['Zach'].notna()]['track_id']
# maggie = df_2024[df_2024['Maggie'].notna()]['track_id']
# jamie = df_2024[df_2024['Jamie'].notna()]['track_id']

# #dict of sets
# sets = {
#     'Bryce': set(bryce),
#     'Zach': set(zach),
#     'Maggie': set(maggie),
#     'Jamie': set(jamie)
# }
    
# venny4py(sets=sets, colors=['#57068c', '#e21833', '#215732', '#25377D'])

overlaps_summary = pd.DataFrame(columns=['2024 Overlaps with Eachother', 'Zach Overlaps with 2023 Self', 'Maggie Overlaps with 2023 Self', 'Jamie Overlaps with 2023 Self', 'Bryce Overlaps with 2023 Self'])
overlaps_summary.loc[0, '2024 Overlaps with Eachother'] = len(overlap_24)
overlaps_summary.loc[0, 'Zach Overlaps with 2023 Self'] = len(zach_overlaps)
overlaps_summary.loc[0, 'Maggie Overlaps with 2023 Self'] = len(maggie_overlaps)
overlaps_summary.loc[0, 'Jamie Overlaps with 2023 Self'] = len(jamie_overlaps)
overlaps_summary.loc[0, 'Bryce Overlaps with 2023 Self'] = len(bryce_overlaps)
overlaps_summary = overlaps_summary.reset_index(drop=True)


newest_songs = df_2024.sort_values(by='Release Date', ascending=False).reset_index(drop=True).head(10)
newest_songs['Listener(s)'] = ""
for rec in newest_songs.index:
    if not pd.isna(newest_songs.loc[rec, 'Zach']):
        newest_songs.loc[rec, 'Listener(s)'] += "Zach "
    if not pd.isna(newest_songs.loc[rec, 'Maggie']):
        newest_songs.loc[rec, 'Listener(s)'] += "Maggie "
    if not pd.isna(newest_songs.loc[rec, 'Jamie']):
        newest_songs.loc[rec, 'Listener(s)'] += "Jamie "
    if not pd.isna(newest_songs.loc[rec, 'Bryce']):
        newest_songs.loc[rec, 'Listener(s)'] += "Bryce "
newest_songs = newest_songs[['Song', 'Artist', 'Listener(s)', 'Release Date']]

oldest_songs = df_2024.sort_values(by='Release Date', ascending=True).reset_index(drop=True).head(10)
oldest_songs['Listener(s)'] = ""
for rec in oldest_songs.index:
    if not pd.isna(oldest_songs.loc[rec, 'Zach']):
        oldest_songs.loc[rec, 'Listener(s)'] += "Zach "
    if not pd.isna(oldest_songs.loc[rec, 'Maggie']):
        oldest_songs.loc[rec, 'Listener(s)'] += "Maggie "
    if not pd.isna(oldest_songs.loc[rec, 'Jamie']):
        oldest_songs.loc[rec, 'Listener(s)'] += "Jamie "
    if not pd.isna(oldest_songs.loc[rec, 'Bryce']):
        oldest_songs.loc[rec, 'Listener(s)'] += "Bryce "
oldest_songs = oldest_songs[['Song', 'Artist', 'Listener(s)', 'Release Date']]


st.set_page_config(layout="wide")
st.title("Spotify Wrapped Unwrapped")

# Main tabs
tab1, tab2, tab3,tab4 = st.tabs(["Overlaps", "Top Artists", "Top Albums", "Song Stats"])

with tab1:
    st.dataframe(overlaps_summary, hide_index=True)
    sub_tab = st.radio("Select Overlap Category:", ["2024 Overlaps", "Zach '23 vs '24", "Bryce '23 vs '24", "Maggie '23 vs '24", "Jamie '23 vs '24", "Influenced"], horizontal=True)
    
    if sub_tab == "2024 Overlaps":
        st.title("2024 Overlaps")
        col1, col2 = st.columns(2)
        with col1:
            st.image("Venn_4.png", width=500)
        with col2:
            st.dataframe(overlap_24, hide_index=True)
    elif sub_tab == "Zach '23 vs '24":
        st.title("Zach's Overlaps")
        st.dataframe(zach_overlaps, hide_index=True)
    elif sub_tab == "Bryce '23 vs '24":
        st.title("Bryce's Overlaps")
        st.dataframe(bryce_overlaps, hide_index=True)
    elif sub_tab == "Maggie '23 vs '24":
        st.title("Maggie's Overlaps")
        st.dataframe(maggie_overlaps, hide_index=True)
    elif sub_tab == "Jamie '23 vs '24":
        st.title("Jamie's Overlaps")
        st.dataframe(jamie_overlaps, hide_index=True)
    elif sub_tab == "Influenced":
        st.title("Influenced")
        st.dataframe(influence, hide_index=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.title("Top Artists")
        st.dataframe(top_artists, hide_index=True)
    with col2:
        st.title("Unique Artists")
        st.write("(including features)")
        st.dataframe(unique_artists_per_listener, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.title("Top Artists by Listener")
        st.dataframe(top_artists_by_listener, hide_index=True)
    with col2:
        st.title("Top Artists Only One Listener")
        st.dataframe(top_artists_only_one_listener, hide_index=True)

        
with tab3:
    st.title("Top Albums")
    st.dataframe(top_albums.head(10), hide_index=True)

with tab4:
    col1, col2,col3 = st.columns(3)
    
    with col1:
        st.title("Most Popular Songs")
        st.dataframe(most_popular, hide_index=True)
        st.caption("Popularity is as of our 100th day")
    with col2:
        st.title("Least Popular Songs")
        st.dataframe(least_popular, hide_index=True)
    with col3:
        st.title("Average Popularity") 
        st.dataframe(avg_popularity, hide_index=True)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.title("Newest Songs")
        st.dataframe(newest_songs, hide_index=True)
    with col2:
        st.title("Oldest Songs")
        st.dataframe(oldest_songs, hide_index=True)
        st.caption("Older songs might only have had release year/month and thus got assigned to the first day of that year/month")
    with col3:
        st.title("Decades")
        st.dataframe(decades, hide_index=True)
        st.dataframe(song_count_2024, hide_index=True)
