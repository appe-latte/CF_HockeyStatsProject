#!/usr/bin/env python3
"""
Hockey Analytics Dashboard - Standalone Version
Can run with or without Streamlit
Usage:
    python hockey_analytics.py                    # Interactive CLI mode
    python hockey_analytics.py --streamlit        # Launch Streamlit app
    python hockey_analytics.py --file data.csv    # Analyze specific file
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Try to import streamlit, but don't fail if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available. Running in CLI mode.")

def load_data(file_path=None):
    """Load data from file or default CSV"""
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    # Try to load default CSV file
    default_files = [
        "olympic_womens_dataset.csv",
        "sample_data.csv",
        "hockey_data.csv",
        "olympic_data.csv"
    ]
    
    for filename in default_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                print(f"Loaded default dataset: {filename}")
                return df
            except Exception as e:
                print(f"Could not load {filename}: {e}")
    
    print("No data file found. Please provide a CSV file.")
    return None

def create_rink_plot():
    """Create a hockey rink plot with proper dimensions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Rink dimensions (standard NHL rink: 200x85 feet)
    rink_length = 200
    rink_width = 85
    
    # Draw rink outline
    ax.plot([0, rink_length, rink_length, 0, 0], 
            [0, 0, rink_width, rink_width, 0], 'k-', linewidth=2)
    
    # Center line
    ax.axvline(x=rink_length/2, color='red', linestyle='--', linewidth=2)
    
    # Blue lines
    ax.axvline(x=25, color='blue', linestyle='-', linewidth=2)
    ax.axvline(x=rink_length-25, color='blue', linestyle='-', linewidth=2)
    
    # Faceoff circles (simplified)
    circle_radius = 15
    # Center faceoff circles
    center_circle = plt.Circle((rink_length/2, rink_width/2), circle_radius, 
                              fill=False, color='red', linewidth=2)
    ax.add_patch(center_circle)
    
    # Neutral zone faceoff circles
    neutral_y_positions = [rink_width/4, 3*rink_width/4]
    for y in neutral_y_positions:
        circle = plt.Circle((rink_length/2, y), circle_radius, 
                           fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
    
    # Offensive zone faceoff circles
    offensive_x_positions = [25, rink_length-25]
    offensive_y_positions = [rink_width/4, 3*rink_width/4]
    for x in offensive_x_positions:
        for y in offensive_y_positions:
            circle = plt.Circle((x, y), circle_radius, 
                               fill=False, color='red', linewidth=2)
            ax.add_patch(circle)
    
    # Goal creases (simplified)
    goal_width = 6
    goal_depth = 4
    # Left goal
    ax.add_patch(plt.Rectangle((0, (rink_width-goal_width)/2), goal_depth, goal_width, 
                               fill=False, color='red', linewidth=2))
    # Right goal
    ax.add_patch(plt.Rectangle((rink_length-goal_depth, (rink_width-goal_width)/2), 
                               goal_depth, goal_width, fill=False, color='red', linewidth=2))
    
    # Set plot limits and labels with proper hockey rink dimensions
    ax.set_xlim(0, rink_length)
    ax.set_ylim(0, rink_width)
    ax.set_aspect('equal')
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    ax.set_title('Hockey Rink Layout', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig, ax

def plot_shot_goal_map(df, team=None, save_path=None):
    """Create shot and goal map visualization"""
    print(f"\n📊 Shot & Goal Map Analysis")
    
    # Filter for shot events
    shot_events = ['Shot', 'Goal']
    shot_df = df[df['Event'].isin(shot_events)].copy()
    
    if shot_df.empty:
        print("No shot events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(shot_df['Team'].unique())
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        team = input("Select team (or press Enter for all teams): ").strip()
        if not team:
            team = teams[0]  # Default to first team
    
    if team not in teams:
        print(f"Team '{team}' not found. Using first available team: {teams[0]}")
        team = teams[0]
    
    # Filter by selected team
    team_shots = shot_df[shot_df['Team'] == team]
    
    if team_shots.empty:
        print(f"No shot data found for {team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Color mapping for shot types
    color_map = {
        'Goal': 'green',
        'Shot': 'blue'
    }
    
    # Plot shots
    for event_type in shot_events:
        if event_type in team_shots['Event'].values:
            event_data = team_shots[team_shots['Event'] == event_type]
            if 'X' in event_data.columns and 'Y' in event_data.columns:
                ax.scatter(event_data['X'], event_data['Y'], 
                          c=color_map.get(event_type, 'gray'),
                          s=50, alpha=0.7, label=event_type)
    
    ax.legend(fontsize=10)
    ax.set_title(f'Shot & Goal Map - {team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_shot_map.png", dpi=300, bbox_inches='tight')
        print(f"Shot map saved to: {save_path}_shot_map.png")
    else:
        plt.show()
    
    # Print statistics
    total_shots = len(team_shots)
    goals = len(team_shots[team_shots['Event'] == 'Goal'])
    shot_percentage = (goals / total_shots * 100) if total_shots > 0 else 0
    shots_on_net = len(team_shots[team_shots['Event'] == 'Shot'])
    
    print(f"\n📈 Shot Statistics for {team}:")
    print(f"   Total Shots: {total_shots}")
    print(f"   Goals: {goals}")
    print(f"   Shot %: {shot_percentage:.1f}%")
    print(f"   Shots on Net: {shots_on_net}")
    
    plt.close()

def plot_passing_network(df, team=None, min_passes=2, save_path=None):
    """Create passing network visualization"""
    print(f"\n🔄 Passing Network Analysis")
    
    # Filter for pass events
    pass_df = df[df['Event'] == 'Play'].copy()
    
    if pass_df.empty:
        print("No pass events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(pass_df['Team'].unique())
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        team = input("Select team (or press Enter for first team): ").strip()
        if not team:
            team = teams[0]
    
    if team not in teams:
        print(f"Team '{team}' not found. Using first available team: {teams[0]}")
        team = teams[0]
    
    # Filter by selected team
    team_passes = pass_df[pass_df['Team'] == team]
    
    if team_passes.empty:
        print(f"No pass data found for {team}.")
        return
    
    # Create pass matrix
    if 'Player' in team_passes.columns and 'Player 2' in team_passes.columns:
        # Count passes between players
        pass_counts = team_passes.groupby(['Player', 'Player 2']).size().reset_index(name='Pass_Count')
        
        # Filter for minimum pass threshold
        pass_counts = pass_counts[pass_counts['Pass_Count'] >= min_passes]
        
        if pass_counts.empty:
            print(f"No pass combinations found with {min_passes} or more passes.")
            return
        
        # Create pivot table for heatmap
        pass_matrix = pass_counts.pivot(index='Player', columns='Player 2', values='Pass_Count').fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pass_matrix, annot=True, cmap='Blues', fmt='g', 
                   cbar_kws={'label': 'Number of Passes', 'fontsize': 12})
        plt.title(f'Passing Network - {team}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Receiver', fontsize=12, fontweight='bold')
        plt.ylabel('Passer', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(f"{save_path}_passing_network.png", dpi=300, bbox_inches='tight')
            print(f"Passing network saved to: {save_path}_passing_network.png")
        else:
            plt.show()
        
        # Print statistics
        total_passes = pass_counts['Pass_Count'].sum()
        unique_players = len(set(pass_counts['Player'].unique()) | set(pass_counts['Player 2'].unique()))
        avg_passes = pass_counts['Pass_Count'].mean()
        
        print(f"\n📈 Passing Statistics for {team}:")
        print(f"   Total Passes: {total_passes}")
        print(f"   Active Players: {unique_players}")
        print(f"   Avg Passes per Connection: {avg_passes:.1f}")
        
        # Show top pass combinations
        print(f"\n🏆 Top Pass Combinations:")
        top_passes = pass_counts.nlargest(5, 'Pass_Count')
        for _, row in top_passes.iterrows():
            print(f"   {row['Player']} → {row['Player 2']}: {row['Pass_Count']} passes")
        
        plt.close()
    
    else:
        print("Player columns not found in the dataset.")

def plot_takeaways(df, team=None, save_path=None):
    """Create takeaways visualization"""
    print(f"\n🎯 Takeaways Analysis")
    
    # Filter for takeaway events
    takeaway_df = df[df['Event'] == 'Takeaway'].copy()
    
    if takeaway_df.empty:
        print("No takeaway events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(takeaway_df['Team'].unique())
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        team = input("Select team (or press Enter for first team): ").strip()
        if not team:
            team = teams[0]
    
    if team not in teams:
        print(f"Team '{team}' not found. Using first available team: {teams[0]}")
        team = teams[0]
    
    # Filter by selected team
    team_takeaways = takeaway_df[takeaway_df['Team'] == team]
    
    if team_takeaways.empty:
        print(f"No takeaway data found for {team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Plot takeaway locations
    if 'X' in team_takeaways.columns and 'Y' in team_takeaways.columns:
        ax.scatter(team_takeaways['X'], team_takeaways['Y'], 
                  c='purple', marker='s', s=100, alpha=0.8, label='Takeaway')
        
        # Annotate player names (limit to avoid overcrowding)
        max_annotations = 15
        for i, (_, row) in enumerate(team_takeaways.iterrows()):
            if i < max_annotations and 'Player' in row and pd.notna(row['Player']):
                ax.annotate(row['Player'], (row['X'], row['Y']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    ax.legend(fontsize=10)
    ax.set_title(f'Takeaways - {team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_takeaways.png", dpi=300, bbox_inches='tight')
        print(f"Takeaways plot saved to: {save_path}_takeaways.png")
    else:
        plt.show()
    
    # Print statistics
    total_takeaways = len(team_takeaways)
    print(f"\n📈 Takeaway Statistics for {team}:")
    print(f"   Total Takeaways: {total_takeaways}")
    
    if 'Player' in team_takeaways.columns:
        top_player = team_takeaways['Player'].value_counts().index[0]
        top_player_takeaways = team_takeaways['Player'].value_counts().iloc[0]
        print(f"   Top Takeaway Player: {top_player} ({top_player_takeaways})")
    
    if 'X' in team_takeaways.columns:
        # Calculate takeaway zones
        rink_length = 200
        offensive_zone = len(team_takeaways[team_takeaways['X'] > rink_length * 0.67])
        print(f"   Offensive Zone Takeaways: {offensive_zone}")
    
    # Show takeaway breakdown by player
    if 'Player' in team_takeaways.columns:
        print(f"\n🏆 Takeaways by Player:")
        player_takeaways = team_takeaways['Player'].value_counts()
        for player, count in player_takeaways.head(5).items():
            print(f"   {player}: {count} takeaways")
    
    plt.close()

def plot_zone_entries(df, team=None, entry_skaters=None, defenders=None, save_path=None):
    """Create zone entries visualization"""
    print(f"\n🚪 Zone Entries Analysis")
    
    # Filter for zone entry events
    entry_df = df[df['Event'] == 'Zone Entry'].copy()
    
    if entry_df.empty:
        print("No zone entry events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(entry_df['Team'].unique())
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        team = input("Select team (or press Enter for first team): ").strip()
        if not team:
            team = teams[0]
    
    if team not in teams:
        print(f"Team '{team}' not found. Using first available team: {teams[0]}")
        team = teams[0]
    
    # Filter by selected team
    team_entries = entry_df[entry_df['Team'] == team]
    
    if team_entries.empty:
        print(f"No zone entry data found for {team}.")
        return
    
    # Entry Skater (Player 1) selection
    if 'Player' in team_entries.columns and entry_skaters is None:
        entry_skaters_list = sorted(team_entries['Player'].unique())
        print(f"Available entry skaters: {', '.join(entry_skaters_list)}")
        skater_input = input("Enter entry skater names to analyze (comma-separated, or press Enter for all): ").strip()
        if skater_input:
            entry_skaters = [p.strip() for p in skater_input.split(',')]
            # Filter to valid skaters
            entry_skaters = [p for p in entry_skaters if p in entry_skaters_list]
            if entry_skaters:
                team_entries = team_entries[team_entries['Player'].isin(entry_skaters)]
                print(f"Analyzing entry skaters: {', '.join(entry_skaters)}")
            else:
                print("No valid entry skaters selected. Analyzing all skaters.")
        else:
            print("Analyzing all entry skaters.")
    
    # Targeted Defender (Player 2) selection
    if 'Player 2' in team_entries.columns and defenders is None:
        defenders_list = sorted(team_entries['Player 2'].unique())
        print(f"Available targeted defenders: {', '.join(defenders_list)}")
        defender_input = input("Enter targeted defender names to analyze (comma-separated, or press Enter for all): ").strip()
        if defender_input:
            defenders = [p.strip() for p in defender_input.split(',')]
            # Filter to valid defenders
            defenders = [p for p in defenders if p in defenders_list]
            if defenders:
                team_entries = team_entries[team_entries['Player 2'].isin(defenders)]
                print(f"Analyzing targeted defenders: {', '.join(defenders)}")
            else:
                print("No valid defenders selected. Analyzing all defenders.")
        else:
            print("Analyzing all targeted defenders.")
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Player color mapping for entry skaters
    if 'Player' in team_entries.columns and entry_skaters and len(entry_skaters) > 1:
        # Assign different colors to each entry skater
        player_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        player_color_map = {}
        
        for i, player in enumerate(entry_skaters):
            player_color_map[player] = player_colors[i % len(player_colors)]
        
        # Plot zone entries by entry skater
        for player in entry_skaters:
            player_data = team_entries[team_entries['Player'] == player]
            if len(player_data) > 0 and 'X' in player_data.columns and 'Y' in player_data.columns:
                ax.scatter(player_data['X'], player_data['Y'], 
                          c=player_color_map[player], s=80, alpha=0.8, 
                          label=f'{player} ({len(player_data)} entries)')
                
                # Annotate some entry locations
                for i, (_, row) in enumerate(player_data.iterrows()):
                    if i < 5:  # Limit annotations per player
                        entry_type = row.get('Detail 1', 'Entry')
                        defender = row.get('Player 2', 'Unknown')
                        ax.annotate(f"{row['Player']}\n{entry_type}\nvs {defender}", 
                                   (row['X'], row['Y']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        # Single player or entry type analysis
        team_colors = {'Team A': 'red', 'Team B': 'green', 'Team C': 'blue', 'Team D': 'orange'}
        team_color = team_colors.get(team, 'purple')
        
        # Color mapping for different entry types within the team color scheme
        if 'Detail 1' in team_entries.columns:
            entry_types = team_entries['Detail 1'].unique()
            
            # Create color variations based on team color
            if team_color == 'red':
                colors = ['darkred', 'red', 'lightcoral', 'indianred', 'crimson']
            elif team_color == 'green':
                colors = ['darkgreen', 'green', 'lightgreen', 'forestgreen', 'limegreen']
            elif team_color == 'blue':
                colors = ['darkblue', 'blue', 'lightblue', 'steelblue', 'skyblue']
            else:
                colors = ['darkorange', 'orange', 'gold', 'darkgoldenrod', 'sandybrown']
            
            # Extend colors if needed
            while len(colors) < len(entry_types):
                colors.extend(colors)
            
            color_map = dict(zip(entry_types, colors[:len(entry_types)]))
            
            # Plot entries by type
            for entry_type in entry_types:
                entry_data = team_entries[team_entries['Detail 1'] == entry_type]
                if len(entry_data) > 0 and 'X' in entry_data.columns and 'Y' in entry_data.columns:
                    ax.scatter(entry_data['X'], entry_data['Y'], 
                              c=[color_map[entry_type]], s=80, alpha=0.8, 
                              label=f'{entry_type} ({len(entry_data)})')
                    
                    # Annotate some entry locations
                    for i, (_, row) in enumerate(entry_data.iterrows()):
                        if i < 8:  # Limit annotations
                            player = row.get('Player', 'Unknown')
                            defender = row.get('Player 2', 'Unknown')
                            ax.annotate(f"{player}\n{entry_type}\nvs {defender}", 
                                       (row['X'], row['Y']), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, alpha=0.8, 
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        else:
            # If no entry types, just plot all entries with team color
            if 'X' in team_entries.columns and 'Y' in team_entries.columns:
                ax.scatter(team_entries['X'], team_entries['Y'], 
                          c=team_color, s=80, alpha=0.8, label=f'{team} Zone Entries')
                
                # Annotate some entry locations
                for i, (_, row) in enumerate(team_entries.iterrows()):
                    if i < 10:  # Limit annotations
                        player = row.get('Player', 'Unknown')
                        defender = row.get('Player 2', 'Unknown')
                        ax.annotate(f"{player}\nvs {defender}", 
                                   (row['X'], row['Y']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
    
    ax.legend(fontsize=10)
    ax.set_title(f'Zone Entries - {team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_zone_entries.png", dpi=300, bbox_inches='tight')
        print(f"Zone entries plot saved to: {save_path}_zone_entries.png")
    else:
        plt.show()
    
    plt.close()
    
    # Entry statistics
    print(f"\n🏒 Zone Entry Statistics:")
    total_entries = len(team_entries)
    print(f"   Total Zone Entries: {total_entries}")
    
    if 'Player' in team_entries.columns and len(team_entries) > 0:
        top_player = team_entries['Player'].value_counts().index[0]
        top_player_entries = team_entries['Player'].value_counts().iloc[0]
        print(f"   Top Entry Player: {top_player} ({top_player_entries} entries)")
    
    if 'Detail 1' in team_entries.columns:
        carried_entries = len(team_entries[team_entries['Detail 1'] == 'Carried'])
        print(f"   Carried Entries: {carried_entries}")
    
    # Show entry type breakdown
    if 'Detail 1' in team_entries.columns and len(team_entries) > 0:
        print(f"\n🏆 Entry Type Breakdown:")
        entry_types = team_entries['Detail 1'].value_counts()
        for entry_type, count in entry_types.items():
            print(f"   {entry_type}: {count}")
    
    # Show detailed entry breakdown
    if 'Player' in team_entries.columns and len(team_entries) > 0:
        print(f"\n🏆 Entry Skater Breakdown:")
        player_entries = team_entries['Player'].value_counts()
        for player, entries in player_entries.items():
            print(f"   {player}: {entries} entries")
        
        # Show detailed breakdown
        print(f"\n🏆 Detailed Zone Entry Breakdown:")
        entry_details = team_entries.groupby(['Player', 'Detail 1']).size().reset_index(name='Entries')
        entry_details = entry_details.sort_values('Entries', ascending=False)
        print(f"{'Player':<20} {'Entry Type':<15} {'Entries':<10}")
        print("-" * 45)
        for _, row in entry_details.iterrows():
            print(f"{row['Player']:<20} {row['Detail 1']:<15} {row['Entries']:<10}")
    
    # Show defender matchup analysis
    if 'Player 2' in team_entries.columns and len(team_entries) > 0:
        print(f"\n🏆 Defender Matchup Analysis:")
        defender_analysis = team_entries.groupby(['Player 2']).size().reset_index(name='Entries Against')
        defender_analysis = defender_analysis.sort_values('Entries Against', ascending=False)
        print(f"{'Defender':<20} {'Entries Against':<15}")
        print("-" * 35)
        for _, row in defender_analysis.iterrows():
            print(f"{row['Player 2']:<20} {row['Entries Against']:<15}")

def plot_faceoff_wins(df, team=None, players=None, save_path=None):
    """Create faceoff wins visualization"""
    print(f"\n🎯 Faceoff Wins Analysis")
    
    # Filter for faceoff events
    faceoff_df = df[df['Event'] == 'Faceoff Win'].copy()
    
    if faceoff_df.empty:
        print("No faceoff win events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(faceoff_df['Team'].unique())
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        team = input("Select team (or press Enter for first team): ").strip()
        if not team:
            team = teams[0]
    
    if team not in teams:
        print(f"Team '{team}' not found. Using first available team: {teams[0]}")
        team = teams[0]
    
    # Filter by selected team
    team_faceoffs = faceoff_df[faceoff_df['Team'] == team]
    
    if team_faceoffs.empty:
        print(f"No faceoff data found for {team}.")
        return
    
    # Player selection
    if 'Player' in team_faceoffs.columns and players is None:
        faceoff_players = sorted(team_faceoffs['Player'].unique())
        print(f"Available players: {', '.join(faceoff_players)}")
        player_input = input("Enter player names to analyze (comma-separated, or press Enter for all): ").strip()
        if player_input:
            players = [p.strip() for p in player_input.split(',')]
            # Filter to valid players
            players = [p for p in players if p in faceoff_players]
            if players:
                team_faceoffs = team_faceoffs[team_faceoffs['Player'].isin(players)]
                print(f"Analyzing players: {', '.join(players)}")
            else:
                print("No valid players selected. Analyzing all players.")
        else:
            print("Analyzing all players.")
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Player color mapping
    if 'Player' in team_faceoffs.columns and players and len(players) > 1:
        # Assign different colors to each player
        player_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        player_color_map = {}
        
        for i, player in enumerate(players):
            player_color_map[player] = player_colors[i % len(player_colors)]
        
        # Plot faceoff locations by player
        for player in players:
            player_data = team_faceoffs[team_faceoffs['Player'] == player]
            if len(player_data) > 0 and 'X' in player_data.columns and 'Y' in player_data.columns:
                ax.scatter(player_data['X'], player_data['Y'], 
                          c=player_color_map[player], marker='x', s=100, alpha=0.8, 
                          label=f'{player} ({len(player_data)} wins)')
                
                # Annotate some faceoff locations
                for i, (_, row) in enumerate(player_data.iterrows()):
                    if i < 5:  # Limit annotations per player
                        ax.annotate(f"{row['Player']}", (row['X'], row['Y']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        # Single player or no player selection - use team color
        team_colors = {'Team A': 'red', 'Team B': 'green', 'Team C': 'blue', 'Team D': 'orange'}
        team_color = team_colors.get(team, 'purple')
        
        if 'X' in team_faceoffs.columns and 'Y' in team_faceoffs.columns:
            ax.scatter(team_faceoffs['X'], team_faceoffs['Y'], 
                      c=team_color, marker='x', s=100, alpha=0.8, label='Faceoff Win')
            
            # Annotate player names (limit to avoid overcrowding)
            max_annotations = 20
            for i, (_, row) in enumerate(team_faceoffs.iterrows()):
                if i < max_annotations and 'Player' in row and pd.notna(row['Player']):
                    ax.annotate(row['Player'], (row['X'], row['Y']), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
    
    ax.legend(fontsize=10)
    ax.set_title(f'Faceoff Wins - {team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_faceoff_wins.png", dpi=300, bbox_inches='tight')
        print(f"Faceoff wins plot saved to: {save_path}_faceoff_wins.png")
    else:
        plt.show()
    
    plt.close()
    
    # Faceoff statistics
    print(f"\n🏒 Faceoff Statistics:")
    total_faceoffs = len(team_faceoffs)
    print(f"   Total Faceoff Wins: {total_faceoffs}")
    
    if 'Player' in team_faceoffs.columns and len(team_faceoffs) > 0:
        top_player = team_faceoffs['Player'].value_counts().index[0]
        top_player_wins = team_faceoffs['Player'].value_counts().iloc[0]
        print(f"   Top Faceoff Winner: {top_player} ({top_player_wins} wins)")
    
    if 'X' in team_faceoffs.columns:
        # Calculate faceoff zones
        rink_length = 200
        offensive_zone = len(team_faceoffs[team_faceoffs['X'] > rink_length * 0.67])
        print(f"   Offensive Zone Wins: {offensive_zone}")
    
    # Show faceoff winners breakdown
    if 'Player' in team_faceoffs.columns and len(team_faceoffs) > 0:
        print(f"\n🏒 Faceoff Winners Breakdown:")
        player_faceoffs = team_faceoffs['Player'].value_counts()
        for player, wins in player_faceoffs.items():
            print(f"   {player}: {wins} wins")
        
        # Show detailed breakdown
        print(f"\n🏒 Detailed Faceoff Breakdown:")
        faceoff_details = team_faceoffs.groupby(['Player']).size().reset_index(name='Wins')
        faceoff_details = faceoff_details.sort_values('Wins', ascending=False)
        print(f"{'Player':<20} {'Wins':<10}")
        print("-" * 30)
        for _, row in faceoff_details.iterrows():
            print(f"{row['Player']:<20} {row['Wins']:<10}")

def plot_penalties(df, team=None, players=None, save_path=None):
    """Create penalties visualization"""
    print(f"\n🚨 Penalties Analysis")
    
    # Filter for penalty events
    penalty_df = df[df['Event'] == 'Penalty Taken'].copy()
    
    if penalty_df.empty:
        print("No penalty events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(penalty_df['Team'].unique())
    
    # Ask if user wants to compare teams
    if team is None:
        print(f"Available teams: {', '.join(teams)}")
        compare_teams = input("Compare multiple teams? (y/n, default n): ").strip().lower()
        
        if compare_teams in ['y', 'yes']:
            print("Enter team names to compare (comma-separated):")
            team_input = input().strip()
            if team_input:
                selected_teams = [t.strip() for t in team_input.split(',')]
                # Filter to valid teams
                selected_teams = [t for t in selected_teams if t in teams]
                if selected_teams:
                    team_penalties = penalty_df[penalty_df['Team'].isin(selected_teams)]
                    team = selected_teams[0]  # For display purposes
                    print(f"Comparing teams: {', '.join(selected_teams)}")
                else:
                    print("No valid teams selected. Using first team.")
                    team = teams[0]
                    team_penalties = penalty_df[penalty_df['Team'] == team]
            else:
                print("No teams entered. Using first team.")
                team = teams[0]
                team_penalties = penalty_df[penalty_df['Team'] == team]
        else:
            # Single team selection
            team = input("Select team (or press Enter for first team): ").strip()
            if not team:
                team = teams[0]
            
            if team not in teams:
                print(f"Team '{team}' not found. Using first available team: {teams[0]}")
                team = teams[0]
            
            # Filter by selected team
            team_penalties = penalty_df[penalty_df['Team'] == team]
    else:
        # Use provided team
        if team not in teams:
            print(f"Team '{team}' not found. Using first available team: {teams[0]}")
            team = teams[0]
        
        # Filter by selected team
        team_penalties = penalty_df[penalty_df['Team'] == team]
    
    if team_penalties.empty:
        print(f"No penalty data found for {team}.")
        return
    
    # Player selection
    if 'Player' in team_penalties.columns:
        penalty_players = sorted(team_penalties['Player'].unique())
        print(f"\nPlayers who took penalties for {team}: {', '.join(penalty_players)}")
        
        if players is None:
            print("Enter player names to analyze (comma-separated, or press Enter for all):")
            player_input = input().strip()
            if player_input:
                selected_players = [p.strip() for p in player_input.split(',')]
                # Filter to only valid players
                selected_players = [p for p in selected_players if p in penalty_players]
                if selected_players:
                    team_penalties = team_penalties[team_penalties['Player'].isin(selected_players)]
                    print(f"Analyzing penalties for: {', '.join(selected_players)}")
                else:
                    print("No valid players selected. Analyzing all players.")
            else:
                print("Analyzing all players.")
        else:
            # Use provided players list
            selected_players = [p for p in players if p in penalty_players]
            if selected_players:
                team_penalties = team_penalties[team_penalties['Player'].isin(selected_players)]
                print(f"Analyzing penalties for: {', '.join(selected_players)}")
            else:
                print("No valid players provided. Analyzing all players.")
    
    # Print statistics
    total_penalties = len(team_penalties)
    print(f"\n📈 Penalty Statistics for {team}:")
    print(f"   Total Penalties: {total_penalties}")
    
    if 'Player' in team_penalties.columns and len(team_penalties) > 0:
        top_player = team_penalties['Player'].value_counts().index[0]
        top_player_penalties = team_penalties['Player'].value_counts().iloc[0]
        print(f"   Most Penalized Player: {top_player} ({top_player_penalties})")
    
    if 'Detail 1' in team_penalties.columns and len(team_penalties) > 0:
        most_common = team_penalties['Detail 1'].value_counts().index[0]
        print(f"   Most Common Infraction: {most_common}")
    
    # Show penalty breakdown by infraction type
    if 'Detail 1' in team_penalties.columns and len(team_penalties) > 0:
        print(f"\n🏆 Penalties by Type:")
        penalty_types = team_penalties['Detail 1'].value_counts()
        for penalty_type, count in penalty_types.items():
            print(f"   {penalty_type}: {count}")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        penalty_types.plot(kind='bar', ax=ax, color='red', alpha=0.7)
        plt.title(f'Penalties by Type - {team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Infraction Type', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Penalties', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(f"{save_path}_penalties.png", dpi=300, bbox_inches='tight')
            print(f"Penalties chart saved to: {save_path}_penalties.png")
        else:
            plt.show()
        
        plt.close()
    
    # Show penalty breakdown by player
    if 'Player' in team_penalties.columns and len(team_penalties) > 0:
        print(f"\n🏆 Penalties by Player:")
        player_penalties = team_penalties['Player'].value_counts()
        for player, count in player_penalties.items():
            print(f"   {player}: {count} penalties")
        
        # Show detailed breakdown
        print(f"\n📊 Detailed Penalty Breakdown:")
        penalty_details = team_penalties.groupby(['Player', 'Detail 1']).size().reset_index(name='Count')
        penalty_details = penalty_details.sort_values('Count', ascending=False)
        for _, row in penalty_details.iterrows():
            print(f"   {row['Player']} - {row['Detail 1']}: {row['Count']} penalties")
        
        # Penalty location analysis
        if 'X' in team_penalties.columns and 'Y' in team_penalties.columns:
            print(f"\n📍 Penalty Location Analysis:")
            
            # Create rink plot for penalty locations
            fig, ax = create_rink_plot()
            
            # Team color mapping
            team_colors = {'Team A': 'red', 'Team B': 'green', 'Team C': 'blue', 'Team D': 'orange'}
            
            # Check if we're comparing multiple teams
            unique_teams = team_penalties['Team'].unique()
            compare_teams = len(unique_teams) > 1
            
            if compare_teams:
                # Plot multiple teams with different colors
                for team_name in unique_teams:
                    team_data = team_penalties[team_penalties['Team'] == team_name]
                    team_color = team_colors.get(team_name, 'purple')
                    
                    if 'Detail 1' in team_data.columns:
                        # Plot by infraction type for each team
                        infraction_types = team_data['Detail 1'].unique()
                        for infraction_type in infraction_types:
                            infraction_data = team_data[team_data['Detail 1'] == infraction_type]
                            if len(infraction_data) > 0:
                                ax.scatter(infraction_data['X'], infraction_data['Y'], 
                                          c=team_color, s=80, alpha=0.8, 
                                          label=f'{team_name} - {infraction_type} ({len(infraction_data)})')
                                
                                # Annotate some key penalties
                                for i, (_, row) in enumerate(infraction_data.iterrows()):
                                    if i < 5:  # Limit annotations per team
                                        ax.annotate(f"{row['Player']}\n{row['Detail 1']}", 
                                                   (row['X'], row['Y']), 
                                                   xytext=(5, 5), textcoords='offset points',
                                                   fontsize=8, alpha=0.8, 
                                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    else:
                        # Plot all penalties for the team
                        ax.scatter(team_data['X'], team_data['Y'], 
                                  c=team_color, s=80, alpha=0.8, label=f'{team_name} Penalties ({len(team_data)})')
            else:
                # Single team analysis with individual player colors
                if 'Player' in team_penalties.columns and 'selected_players' in locals() and len(selected_players) > 1:
                    # Assign different colors to each player
                    player_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    player_color_map = {}
                    
                    for i, player in enumerate(selected_players):
                        player_color_map[player] = player_colors[i % len(player_colors)]
                    
                    # Plot penalties by player
                    for player in selected_players:
                        player_data = team_penalties[team_penalties['Player'] == player]
                        if len(player_data) > 0 and 'X' in player_data.columns and 'Y' in player_data.columns:
                            ax.scatter(player_data['X'], player_data['Y'], 
                                      c=player_color_map[player], s=80, alpha=0.8, 
                                      label=f'{player} ({len(player_data)} penalties)')
                            
                            # Annotate some penalty locations
                            for i, (_, row) in enumerate(player_data.iterrows()):
                                if i < 5:  # Limit annotations per player
                                    infraction_type = row.get('Detail 1', 'Unknown')
                                    ax.annotate(f"{row['Player']}\n{infraction_type}", 
                                               (row['X'], row['Y']), 
                                               xytext=(5, 5), textcoords='offset points',
                                               fontsize=8, alpha=0.8, 
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                else:
                    # Single player or infraction type analysis
                    team_color = team_colors.get(team, 'purple')
                    
                    # Color mapping for different infraction types within the team color scheme
                    if 'Detail 1' in team_penalties.columns:
                        infraction_types = team_penalties['Detail 1'].unique()
                        
                        # Create color variations based on team color
                        if team_color == 'red':
                            colors = ['darkred', 'red', 'lightcoral', 'indianred', 'crimson']
                        elif team_color == 'green':
                            colors = ['darkgreen', 'green', 'lightgreen', 'forestgreen', 'limegreen']
                        elif team_color == 'blue':
                            colors = ['darkblue', 'blue', 'lightblue', 'steelblue', 'skyblue']
                        else:
                            colors = ['darkorange', 'orange', 'gold', 'darkgoldenrod', 'sandybrown']
                        
                        # Extend colors if needed
                        while len(colors) < len(infraction_types):
                            colors.extend(colors)
                        
                        color_map = dict(zip(infraction_types, colors[:len(infraction_types)]))
                        
                        # Plot penalties by infraction type
                        for infraction_type in infraction_types:
                            infraction_data = team_penalties[team_penalties['Detail 1'] == infraction_type]
                            if len(infraction_data) > 0:
                                ax.scatter(infraction_data['X'], infraction_data['Y'], 
                                          c=[color_map[infraction_type]], s=80, alpha=0.8, 
                                          label=f'{infraction_type} ({len(infraction_data)})')
                                
                                # Annotate player names for key penalties
                                for i, (_, row) in enumerate(infraction_data.iterrows()):
                                    if i < 10:  # Limit annotations to avoid overcrowding
                                        ax.annotate(f"{row['Player']}\n{row['Detail 1']}", 
                                                   (row['X'], row['Y']), 
                                                   xytext=(5, 5), textcoords='offset points',
                                                   fontsize=8, alpha=0.8, 
                                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    else:
                        # If no infraction types, just plot all penalties with team color
                        ax.scatter(team_penalties['X'], team_penalties['Y'], 
                                  c=team_color, s=80, alpha=0.8, label=f'{team} Penalties')
                        
                        # Annotate player names
                        for i, (_, row) in enumerate(team_penalties.iterrows()):
                            if i < 15:  # Limit annotations
                                ax.annotate(row['Player'], (row['X'], row['Y']), 
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.8)
            
            ax.legend(fontsize=10)
            
            if compare_teams:
                ax.set_title(f'Penalty Locations - Team Comparison', fontsize=16, fontweight='bold', pad=20)
            else:
                ax.set_title(f'Penalty Locations - {team}', fontsize=16, fontweight='bold', pad=20)
                
            ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
            
            # Ensure proper axis limits
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 85)
            
            # Save or show plot
            if save_path:
                plt.savefig(f"{save_path}_penalty_locations.png", dpi=300, bbox_inches='tight')
                print(f"Penalty locations plot saved to: {save_path}_penalty_locations.png")
            else:
                plt.show()
            
            plt.close()
            
            # Zone analysis
            print(f"\n🏒 Penalty Zone Analysis:")
            
            if compare_teams:
                # Show zone analysis for each team
                for team_name in unique_teams:
                    team_data = team_penalties[team_penalties['Team'] == team_name]
                    print(f"\n   {team_name}:")
                    offensive_zone = len(team_data[team_data['X'] > 133])
                    neutral_zone = len(team_data[(team_data['X'] >= 67) & (team_data['X'] <= 133)])
                    defensive_zone = len(team_data[team_data['X'] < 67])
                    
                    print(f"     Offensive Zone: {offensive_zone}")
                    print(f"     Neutral Zone: {neutral_zone}")
                    print(f"     Defensive Zone: {defensive_zone}")
            else:
                # Single team zone analysis
                offensive_zone = len(team_penalties[team_penalties['X'] > 133])  # 2/3 of rink
                neutral_zone = len(team_penalties[(team_penalties['X'] >= 67) & (team_penalties['X'] <= 133)])
                defensive_zone = len(team_penalties[team_penalties['X'] < 67])
                
                print(f"   Offensive Zone Penalties: {offensive_zone}")
                print(f"   Neutral Zone Penalties: {neutral_zone}")
                print(f"   Defensive Zone Penalties: {defensive_zone}")
        
        # Comprehensive penalty data with sorting
        print(f"\n📋 Complete Penalty Data:")
        
        # Create comprehensive dataframe with all relevant information
        penalty_data = team_penalties.copy()
        
        # Add zone information
        if 'X' in penalty_data.columns:
            penalty_data['Zone'] = penalty_data['X'].apply(
                lambda x: 'Offensive' if x > 133 else ('Neutral' if x >= 67 else 'Defensive')
            )
        
        # Add time information if available
        if 'Time' in penalty_data.columns:
            penalty_data['Time'] = penalty_data['Time'].fillna('Unknown')
        else:
            penalty_data['Time'] = 'Unknown'
        
        if 'Period' in penalty_data.columns:
            penalty_data['Period'] = penalty_data['Period'].fillna('Unknown')
        else:
            penalty_data['Period'] = 'Unknown'
        
        # Select and reorder columns for display
        display_columns = ['Team', 'Player', 'Detail 1', 'Zone', 'Period', 'Time']
        if 'X' in penalty_data.columns:
            display_columns.extend(['X', 'Y'])
        
        # Filter to only columns that exist
        display_columns = [col for col in display_columns if col in penalty_data.columns]
        
        # Create the display dataframe
        display_df = penalty_data[display_columns].copy()
        
        # Sort by team, player and infraction type by default
        if 'Team' in display_df.columns and 'Player' in display_df.columns and 'Detail 1' in display_df.columns:
            display_df = display_df.sort_values(['Team', 'Player', 'Detail 1'])
        elif 'Player' in display_df.columns and 'Detail 1' in display_df.columns:
            display_df = display_df.sort_values(['Player', 'Detail 1'])
        
        # Display the data
        if 'Team' in display_df.columns:
            print(f"\n{'Team':<10} {'Player':<15} {'Infraction':<15} {'Zone':<12} {'Period':<8} {'Time':<10} {'X':<6} {'Y':<6}")
            print("-" * 90)
            for _, row in display_df.iterrows():
                team = str(row.get('Team', 'Unknown'))[:9]
                player = str(row.get('Player', 'Unknown'))[:14]
                infraction = str(row.get('Detail 1', 'Unknown'))[:14]
                zone = str(row.get('Zone', 'Unknown'))[:11]
                period = str(row.get('Period', 'Unknown'))[:7]
                time = str(row.get('Time', 'Unknown'))[:9]
                x = str(row.get('X', 'N/A'))[:5]
                y = str(row.get('Y', 'N/A'))[:5]
                print(f"{team:<10} {player:<15} {infraction:<15} {zone:<12} {period:<8} {time:<10} {x:<6} {y:<6}")
        else:
            print(f"\n{'Player':<15} {'Infraction':<15} {'Zone':<12} {'Period':<8} {'Time':<10} {'X':<6} {'Y':<6}")
            print("-" * 80)
            for _, row in display_df.iterrows():
                player = str(row.get('Player', 'Unknown'))[:14]
                infraction = str(row.get('Detail 1', 'Unknown'))[:14]
                zone = str(row.get('Zone', 'Unknown'))[:11]
                period = str(row.get('Period', 'Unknown'))[:7]
                time = str(row.get('Time', 'Unknown'))[:9]
                x = str(row.get('X', 'N/A'))[:5]
                y = str(row.get('Y', 'N/A'))[:5]
                print(f"{player:<15} {infraction:<15} {zone:<12} {period:<8} {time:<10} {x:<6} {y:<6}")
        
        # Save detailed data to CSV if save_path provided
        if save_path:
            csv_filename = f"{save_path}_penalty_details.csv"
            display_df.to_csv(csv_filename, index=False)
            print(f"\n📄 Detailed penalty data saved to: {csv_filename}")

def show_dataset_info(df):
    """Display basic dataset information"""
    print(f"\n📊 Dataset Information:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Column names: {', '.join(df.columns)}")
    
    if 'Team' in df.columns:
        teams = sorted(df['Team'].unique())
        print(f"   Teams: {len(teams)}")
        print(f"   Team names: {', '.join(teams)}")
    
    if 'Event' in df.columns:
        events = df['Event'].value_counts()
        print(f"   Event types: {len(events)}")
        print(f"   Top 5 events:")
        for event, count in events.head(5).items():
            print(f"     {event}: {count}")

def interactive_menu(df):
    """Interactive menu for CLI mode"""
    while True:
        print(f"\n" + "="*50)
        print(f"🏒 HOCKEY ANALYTICS DASHBOARD")
        print(f"="*50)
        print(f"1. Shot & Goal Map")
        print(f"2. Passing Network")
        print(f"3. Takeaways")
        print(f"4. Zone Entries")
        print(f"5. Faceoff Wins")
        print(f"6. Penalties")
        print(f"7. Dataset Information")
        print(f"8. Save All Plots")
        print(f"0. Exit")
        
        choice = input(f"\nSelect option (0-8): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            plot_shot_goal_map(df)
        elif choice == '2':
            min_passes = input("Minimum passes to display (default 2): ").strip()
            min_passes = int(min_passes) if min_passes.isdigit() else 2
            plot_passing_network(df, min_passes=min_passes)
        elif choice == '3':
            plot_takeaways(df)
        elif choice == '4':
            plot_zone_entries(df)
        elif choice == '5':
            plot_faceoff_wins(df)
        elif choice == '6':
            plot_penalties(df)
        elif choice == '7':
            show_dataset_info(df)
        elif choice == '8':
            save_path = input("Enter base filename for saving plots: ").strip()
            if save_path:
                plot_shot_goal_map(df, save_path=save_path)
                plot_passing_network(df, save_path=save_path)
                plot_takeaways(df, save_path=save_path)
                plot_zone_entries(df, save_path=save_path)
                plot_faceoff_wins(df, save_path=save_path)
                plot_penalties(df, save_path=save_path)
                print(f"All plots saved with base filename: {save_path}")
        else:
            print("Invalid option. Please try again.")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Hockey Analytics Dashboard')
    parser.add_argument('--streamlit', action='store_true', 
                       help='Launch Streamlit app (if available)')
    parser.add_argument('--file', type=str, 
                       help='Path to CSV data file')
    parser.add_argument('--team', type=str, 
                       help='Team to analyze')
    parser.add_argument('--players', type=str, 
                       help='Comma-separated list of players to analyze (for penalties)')
    parser.add_argument('--save', type=str, 
                       help='Base filename for saving plots')
    parser.add_argument('--shots', action='store_true', 
                       help='Generate shot map')
    parser.add_argument('--passing', action='store_true', 
                       help='Generate passing network')
    parser.add_argument('--takeaways', action='store_true', 
                       help='Generate takeaways analysis')
    parser.add_argument('--zone-entries', action='store_true', 
                       help='Generate zone entries analysis')
    parser.add_argument('--faceoffs', action='store_true', 
                       help='Generate faceoff wins analysis')
    parser.add_argument('--penalties', action='store_true', 
                       help='Generate penalties analysis')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.file)
    if df is None:
        return
    
    # Show dataset info
    show_dataset_info(df)
    
    # Handle different modes
    if args.streamlit and STREAMLIT_AVAILABLE:
        print("Launching Streamlit app...")
        os.system(f"streamlit run app.py")
    elif args.streamlit and not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Running in CLI mode.")
        interactive_menu(df)
    elif any([args.shots, args.passing, args.takeaways, args.zone_entries, args.faceoffs, args.penalties]):
        # Parse players list if provided
        players_list = None
        if args.players:
            players_list = [p.strip() for p in args.players.split(',')]
        
        # Generate specific plots
        if args.shots:
            plot_shot_goal_map(df, team=args.team, save_path=args.save)
        if args.passing:
            plot_passing_network(df, team=args.team, save_path=args.save)
        if args.takeaways:
            plot_takeaways(df, team=args.team, save_path=args.save)
        if args.zone_entries:
            plot_zone_entries(df, team=args.team, entry_skaters=players_list, defenders=players_list, save_path=args.save)
        if args.faceoffs:
            plot_faceoff_wins(df, team=args.team, players=players_list, save_path=args.save)
        if args.penalties:
            plot_penalties(df, team=args.team, players=players_list, save_path=args.save)
    else:
        # Interactive mode
        interactive_menu(df)

if __name__ == "__main__":
    main() 