import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Women's Hockey Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Load data from uploaded file or default CSV"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded data from uploaded file: {uploaded_file.name}")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    
    # Try to load default CSV file
    try:
        df = pd.read_csv("olympic_womens_dataset.csv")
        st.info("Loaded default dataset: olympic_womens_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("No data file found. Please upload a CSV file or ensure 'olympic_womens_dataset.csv' is in the working directory.")
        return None
    except Exception as e:
        st.error(f"Error loading default file: {e}")
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

def validate_coordinates(df, event_type="events"):
    """Validate and display coordinate information for the dataset"""
    # Check for different possible coordinate column names
    x_col = None
    y_col = None
    
    if 'X' in df.columns and 'Y' in df.columns:
        x_col, y_col = 'X', 'Y'
    elif 'X Coordinate' in df.columns and 'Y Coordinate' in df.columns:
        x_col, y_col = 'X Coordinate', 'Y Coordinate'
    else:
        st.warning(f"No X,Y coordinates found in the dataset for {event_type}.")
        return False
    
    # Check for missing coordinates
    missing_coords = df[[x_col, y_col]].isnull().sum()
    if missing_coords[x_col] > 0 or missing_coords[y_col] > 0:
        st.warning(f"Some {event_type} are missing coordinate data: {missing_coords[x_col]} missing X, {missing_coords[y_col]} missing Y")
    
    # Display coordinate range
    x_range = f"{df[x_col].min():.1f} - {df[x_col].max():.1f}"
    y_range = f"{df[y_col].min():.1f} - {df[y_col].max():.1f}"
    
    st.info(f"Coordinate ranges - X: {x_range}, Y: {y_range}")
    
    return True

def plot_coordinates_on_rink(df, ax, color='blue', marker='o', size=50, alpha=0.7, label='Events'):
    """Helper function to plot coordinates on the rink with proper validation"""
    # Check for different possible coordinate column names
    x_col = None
    y_col = None
    
    if 'X' in df.columns and 'Y' in df.columns:
        x_col, y_col = 'X', 'Y'
    elif 'X Coordinate' in df.columns and 'Y Coordinate' in df.columns:
        x_col, y_col = 'X Coordinate', 'Y Coordinate'
    else:
        st.warning("No X,Y coordinates found in the dataset.")
        return None
    
    # Filter out rows with missing coordinates
    valid_coords = df.dropna(subset=[x_col, y_col])
    
    if valid_coords.empty:
        st.warning("No valid coordinate data found.")
        return None
    
    # Plot the coordinates
    ax.scatter(valid_coords[x_col], valid_coords[y_col], 
              c=color, marker=marker, s=size, alpha=alpha, label=label)
    
    return valid_coords

def get_coordinate_columns(df):
    """Helper function to get the correct coordinate column names"""
    if 'X' in df.columns and 'Y' in df.columns:
        return 'X', 'Y'
    elif 'X Coordinate' in df.columns and 'Y Coordinate' in df.columns:
        return 'X Coordinate', 'Y Coordinate'
    else:
        return None, None

def plot_all_events_map(df):
    """Create a comprehensive map showing all events with coordinates"""
    st.markdown('<div class="section-header">All Events Map</div>', unsafe_allow_html=True)
    
    # Validate coordinates for the entire dataset
    if not validate_coordinates(df, "all events"):
        st.warning("Cannot create comprehensive map without coordinate data.")
        return
    
    # Team selection
    teams = sorted(df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="all_events_team")
    
    # Filter by selected team
    team_data = df[df['Team'] == selected_team].copy()
    
    if team_data.empty:
        st.warning(f"No data found for {selected_team}.")
        return
    
    # Event type selection
    all_events = sorted(team_data['Event'].unique())
    selected_events = st.multiselect(
        "Select Event Types to Display:",
        options=all_events,
        default=all_events,
        help="Choose which event types to show on the map"
    )
    
    if not selected_events:
        st.warning("Please select at least one event type.")
        return
    
    # Filter by selected events
    filtered_data = team_data[team_data['Event'].isin(selected_events)]
    
    if filtered_data.empty:
        st.warning(f"No data found for selected events for {selected_team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Color mapping for different event types
    event_colors = {
        'Shot': 'blue',
        'Goal': 'green',
        'Play': 'orange',
        'Takeaway': 'purple',
        'Puck Recovery': 'brown',
        'Dump In': 'lightgreen',
        'Dump Out': 'red',
        'Zone Entry': 'cyan',
        'Faceoff Win': 'magenta',
        'Penalty Taken': 'darkred'
    }
    
    # Plot each event type
    for event_type in selected_events:
        event_data = filtered_data[filtered_data['Event'] == event_type]
        if not event_data.empty:
            color = event_colors.get(event_type, 'gray')
            valid_data = plot_coordinates_on_rink(
                event_data, ax, 
                color=color, size=40, alpha=0.7, 
                label=f'{event_type} ({len(event_data)})'
            )
            
            # Add annotations for key events (limit to avoid overcrowding)
            if valid_data is not None and len(valid_data) <= 20:  # Only annotate if not too many events
                x_col, y_col = get_coordinate_columns(valid_data)
                if x_col and y_col:
                    for i, (_, row) in enumerate(valid_data.iterrows()):
                        if i < 5:  # Limit annotations per event type
                            player = row.get('Player', 'Unknown')
                            ax.annotate(f"{player}\n{event_type}", 
                                       (row[x_col], row[y_col]), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=7, alpha=0.8, 
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_title(f'All Events Map - {selected_team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Event statistics
    st.subheader("Event Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = len(filtered_data)
        st.metric("Total Events", total_events)
    
    with col2:
        unique_players = filtered_data['Player'].nunique()
        st.metric("Active Players", unique_players)
    
    with col3:
        x_col, _ = get_coordinate_columns(filtered_data)
        if x_col:
            offensive_zone = len(filtered_data[filtered_data[x_col] > 133])
            st.metric("Offensive Zone Events", offensive_zone)
    
    with col4:
        x_col, _ = get_coordinate_columns(filtered_data)
        if x_col:
            defensive_zone = len(filtered_data[filtered_data[x_col] < 67])
            st.metric("Defensive Zone Events", defensive_zone)
    
    # Event breakdown
    st.subheader("Event Breakdown")
    event_counts = filtered_data['Event'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    event_counts.plot(kind='bar', ax=ax, color=[event_colors.get(event, 'gray') for event in event_counts.index])
    plt.title(f'Event Breakdown - {selected_team}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Event Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Events', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Zone analysis
    x_col, _ = get_coordinate_columns(filtered_data)
    if x_col:
        st.subheader("Zone Analysis")
        
        # Define zones
        filtered_data['Zone'] = filtered_data[x_col].apply(
            lambda x: 'Offensive' if x > 133 else ('Neutral' if x >= 67 else 'Defensive')
        )
        
        zone_counts = filtered_data['Zone'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            zone_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            plt.title(f'Events by Zone - {selected_team}', fontsize=14, fontweight='bold', pad=20)
            plt.ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Zone breakdown by event type
            zone_event_counts = filtered_data.groupby(['Zone', 'Event']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            zone_event_counts.plot(kind='bar', ax=ax, stacked=True)
            plt.title(f'Events by Zone and Type - {selected_team}', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Zone', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Events', fontsize=12, fontweight='bold')
            plt.xticks(rotation=0, fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)

def plot_shot_goal_map(df):
    """Create shot and goal map visualization"""
    st.markdown('<div class="section-header">Shot & Goal Map</div>', unsafe_allow_html=True)
    
    # Filter for shot events
    shot_events = ['Shot', 'Goal']
    shot_df = df[df['Event'].isin(shot_events)].copy()
    
    if shot_df.empty:
        st.warning("No shot events found in the dataset.")
        return
    
    # Validate coordinates
    if not validate_coordinates(shot_df, "shot events"):
        st.warning("Cannot create shot map without coordinate data.")
        return
    
    # Team selection
    teams = sorted(shot_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="shot_team")
    
    # Event type filter
    st.subheader("Event Filter")
    event_filter = st.radio(
        "Select events to display:",
        options=["Both", "Shots Only", "Goals Only"],
        horizontal=True,
        help="Choose which types of events to show on the map"
    )
    
    # Filter by selected team
    team_shots = shot_df[shot_df['Team'] == selected_team]
    
    if team_shots.empty:
        st.warning(f"No shot data found for {selected_team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Color mapping for shot types
    color_map = {
        'Goal': 'green',
        'Shot': 'blue'
    }
    
    # Determine which events to plot based on filter
    events_to_plot = []
    if event_filter == "Both":
        events_to_plot = shot_events
    elif event_filter == "Shots Only":
        events_to_plot = ['Shot']
    elif event_filter == "Goals Only":
        events_to_plot = ['Goal']
    
    # Plot selected events with coordinate validation
    for event_type in events_to_plot:
        if event_type in team_shots['Event'].values:
            event_data = team_shots[team_shots['Event'] == event_type]
            valid_data = plot_coordinates_on_rink(
                event_data, ax, 
                color=color_map.get(event_type, 'gray'),
                size=50, alpha=0.7, label=event_type
            )
            
            # Add annotations for goals
            if event_type == 'Goal' and valid_data is not None:
                x_col, y_col = get_coordinate_columns(valid_data)
                if x_col and y_col:
                    for i, (_, row) in enumerate(valid_data.iterrows()):
                        if i < 10:  # Limit annotations
                            player = row.get('Player', 'Unknown')
                            ax.annotate(f"GOAL\n{player}", (row[x_col], row[y_col]), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, alpha=0.8, 
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.legend(fontsize=10)
    
    # Update title based on filter
    if event_filter == "Both":
        title = f'Shot & Goal Map - {selected_team}'
    elif event_filter == "Shots Only":
        title = f'Shot Map - {selected_team}'
    elif event_filter == "Goals Only":
        title = f'Goal Map - {selected_team}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Shot statistics based on filter
    col1, col2, col3, col4 = st.columns(4)
    
    # Filter data based on event filter
    if event_filter == "Both":
        filtered_data = team_shots
    elif event_filter == "Shots Only":
        filtered_data = team_shots[team_shots['Event'] == 'Shot']
    elif event_filter == "Goals Only":
        filtered_data = team_shots[team_shots['Event'] == 'Goal']
    
    with col1:
        total_events = len(filtered_data)
        st.metric("Total Events", total_events)
    
    with col2:
        if event_filter == "Both":
            goals = len(team_shots[team_shots['Event'] == 'Goal'])
            st.metric("Goals", goals)
        elif event_filter == "Goals Only":
            st.metric("Goals", total_events)
        else:
            st.metric("Goals", 0)
    
    with col3:
        if event_filter == "Both":
            total_shots = len(team_shots)
            goals = len(team_shots[team_shots['Event'] == 'Goal'])
            shot_percentage = (goals / total_shots * 100) if total_shots > 0 else 0
            st.metric("Shot %", f"{shot_percentage:.1f}%")
        elif event_filter == "Goals Only":
            st.metric("Shot %", "100.0%")
        else:
            st.metric("Shot %", "0.0%")
    
    with col4:
        if event_filter == "Both":
            shots_on_net = len(team_shots[team_shots['Event'] == 'Shot'])
            st.metric("Shots on Net", shots_on_net)
        elif event_filter == "Shots Only":
            st.metric("Shots on Net", total_events)
        else:
            st.metric("Shots on Net", 0)

def plot_passing_network(df):
    """Create passing network visualization"""
    st.markdown('<div class="section-header">Passing Network</div>', unsafe_allow_html=True)
    
    # Filter for pass events
    pass_df = df[df['Event'] == 'Play'].copy()
    
    if pass_df.empty:
        st.warning("No pass events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(pass_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="pass_team")
    
    # Filter by selected team
    team_passes = pass_df[pass_df['Team'] == selected_team]
    
    if team_passes.empty:
        st.warning(f"No pass data found for {selected_team}.")
        return
    
    # Create pass matrix
    if 'Player' in team_passes.columns and 'Player 2' in team_passes.columns:
        # Count passes between players
        pass_counts = team_passes.groupby(['Player', 'Player 2']).size().reset_index(name='Pass_Count')
        
        # Filter for minimum pass threshold
        min_passes = st.slider("Minimum number of passes to display:", 1, 10, 2)
        pass_counts = pass_counts[pass_counts['Pass_Count'] >= min_passes]
        
        if pass_counts.empty:
            st.warning(f"No pass combinations found with {min_passes} or more passes.")
            return
        
        # Create pivot table for heatmap
        pass_matrix = pass_counts.pivot(index='Player', columns='Player 2', values='Pass_Count').fillna(0)
        
        # Create heatmap with error handling
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(pass_matrix, annot=True, cmap='Blues', fmt='g', 
                       cbar_kws={'label': 'Number of Passes'})
            plt.title(f'Passing Network - {selected_team}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Receiver', fontsize=12, fontweight='bold')
            plt.ylabel('Passer', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            st.info("This might be due to insufficient data or compatibility issues.")
            return
        
        # Pass statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_passes = pass_counts['Pass_Count'].sum()
            st.metric("Total Passes", total_passes)
        
        with col2:
            unique_players = len(set(pass_counts['Player'].unique()) | set(pass_counts['Player 2'].unique()))
            st.metric("Active Players", unique_players)
        
        with col3:
            avg_passes = pass_counts['Pass_Count'].mean()
            st.metric("Avg Passes per Connection", f"{avg_passes:.1f}")
        
        # Show top pass combinations
        st.subheader("Top Pass Combinations")
        top_passes = pass_counts.nlargest(10, 'Pass_Count')
        st.dataframe(top_passes, use_container_width=True)
    
    else:
        st.warning("Player columns not found in the dataset.")

def plot_takeaways(df):
    """Create takeaways visualization"""
    st.markdown('<div class="section-header">Takeaways</div>', unsafe_allow_html=True)
    
    # Filter for takeaway events
    takeaway_df = df[df['Event'] == 'Takeaway'].copy()
    
    if takeaway_df.empty:
        st.warning("No takeaway events found in the dataset.")
        return
    
    # Validate coordinates
    if not validate_coordinates(takeaway_df, "takeaway events"):
        st.warning("Cannot create takeaway map without coordinate data.")
        return
    
    # Team selection
    teams = sorted(takeaway_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="takeaway_team")
    
    # Filter by selected team
    team_takeaways = takeaway_df[takeaway_df['Team'] == selected_team]
    
    if team_takeaways.empty:
        st.warning(f"No takeaway data found for {selected_team}.")
        return
    
    # Player selection filter
    st.subheader("Player Filter")
    if 'Player' in team_takeaways.columns:
        # Get all players who had takeaways
        takeaway_players = sorted(team_takeaways['Player'].unique())
        
        # Add "All Players" option
        player_options = ["All Players"] + takeaway_players
        
        selected_player = st.selectbox(
            "Select Player:",
            options=player_options,
            help="Choose a specific player to analyze or 'All Players' to see team data"
        )
        
        # Filter by selected player
        if selected_player != "All Players":
            team_takeaways = team_takeaways[team_takeaways['Player'] == selected_player]
            if team_takeaways.empty:
                st.warning(f"No takeaway data found for {selected_player}.")
                return
    else:
        st.warning("Player column not found in the dataset.")
        return
    
    if team_takeaways.empty:
        st.warning(f"No takeaway data found for {selected_team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Plot takeaway locations with coordinate validation
    if selected_player == "All Players":
        # Plot all takeaways with team color
        valid_data = plot_coordinates_on_rink(
            team_takeaways, ax, 
            color='purple', marker='s', size=100, alpha=0.8, label='Takeaway'
        )
    else:
        # Plot individual player takeaways with distinct color
        valid_data = plot_coordinates_on_rink(
            team_takeaways, ax, 
            color='red', marker='s', size=120, alpha=0.9, label=f'{selected_player} Takeaways'
        )
    
    # Annotate player names if valid data exists
    if valid_data is not None:
        x_col, y_col = get_coordinate_columns(valid_data)
        if x_col and y_col:
            if selected_player == "All Players":
                # Annotate player names for team view
                max_annotations = 15
                for i, (_, row) in enumerate(valid_data.iterrows()):
                    if i < max_annotations and 'Player' in row and pd.notna(row['Player']):
                        ax.annotate(row['Player'], (row[x_col], row[y_col]), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
            else:
                # Annotate takeaway details for individual player
                for i, (_, row) in enumerate(valid_data.iterrows()):
                    if i < 10:  # Limit annotations for individual player
                        # Add takeaway number or other details
                        takeaway_num = i + 1
                        ax.annotate(f"#{takeaway_num}", (row[x_col], row[y_col]), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=9, alpha=0.8, 
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.legend(fontsize=10)
    
    # Update title based on player selection
    if selected_player == "All Players":
        title = f'Takeaways - {selected_team}'
    else:
        title = f'Takeaways - {selected_player} ({selected_team})'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Takeaway statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_takeaways = len(team_takeaways)
        if selected_player == "All Players":
            st.metric("Total Takeaways", total_takeaways)
        else:
            st.metric(f"{selected_player} Takeaways", total_takeaways)
    
    with col2:
        if selected_player == "All Players" and 'Player' in team_takeaways.columns:
            top_player = team_takeaways['Player'].value_counts().index[0]
            top_player_takeaways = team_takeaways['Player'].value_counts().iloc[0]
            st.metric("Top Takeaway Player", f"{top_player} ({top_player_takeaways})")
        elif selected_player != "All Players":
            st.metric("Player", selected_player)
        else:
            st.metric("Player", "N/A")
    
    with col3:
        x_col, _ = get_coordinate_columns(team_takeaways)
        if x_col:
            # Calculate takeaway zones
            rink_length = 200
            offensive_zone = len(team_takeaways[team_takeaways[x_col] > rink_length * 0.67])
            neutral_zone = len(team_takeaways[(team_takeaways[x_col] >= rink_length * 0.33) & 
                                            (team_takeaways[x_col] <= rink_length * 0.67)])
            defensive_zone = len(team_takeaways[team_takeaways[x_col] < rink_length * 0.33])
            
            if selected_player == "All Players":
                st.metric("Offensive Zone Takeaways", offensive_zone)
            else:
                st.metric("Offensive Zone Takeaways", offensive_zone)
    
    # Show takeaway breakdown by player
    if 'Player' in team_takeaways.columns:
        if selected_player == "All Players":
            st.subheader("Takeaway Breakdown by Player")
            player_takeaways = team_takeaways['Player'].value_counts()
            
            # Create a proper bar chart with matplotlib for better formatting
            fig, ax = plt.subplots(figsize=(10, 6))
            player_takeaways.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
            plt.title(f'Takeaways by Player - {selected_team}', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Player', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Takeaways', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            # Show individual player statistics
            st.subheader(f"{selected_player} Takeaway Details")
            
            # Zone breakdown for individual player
            x_col, _ = get_coordinate_columns(team_takeaways)
            if x_col:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    offensive_zone = len(team_takeaways[team_takeaways[x_col] > 133])
                    st.metric("Offensive Zone", offensive_zone)
                
                with col2:
                    neutral_zone = len(team_takeaways[(team_takeaways[x_col] >= 67) & (team_takeaways[x_col] <= 133)])
                    st.metric("Neutral Zone", neutral_zone)
                
                with col3:
                    defensive_zone = len(team_takeaways[team_takeaways[x_col] < 67])
                    st.metric("Defensive Zone", defensive_zone)

def plot_puck_recoveries(df):
    """Create puck recoveries visualization"""
    st.markdown('<div class="section-header">Puck Recoveries</div>', unsafe_allow_html=True)
    
    # Filter for puck recovery events
    recovery_df = df[df['Event'] == 'Puck Recovery'].copy()
    
    if recovery_df.empty:
        st.warning("No puck recovery events found in the dataset.")
        return
    
    # Validate coordinates
    if not validate_coordinates(recovery_df, "puck recovery events"):
        st.warning("Cannot create puck recovery map without coordinate data.")
        return
    
    # Team selection
    teams = sorted(recovery_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="recovery_team")
    
    # Filter by selected team
    team_recoveries = recovery_df[recovery_df['Team'] == selected_team]
    
    if team_recoveries.empty:
        st.warning(f"No puck recovery data found for {selected_team}.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Plot recovery locations with coordinate validation
    valid_data = plot_coordinates_on_rink(
        team_recoveries, ax, 
        color='orange', marker='o', size=80, alpha=0.8, label='Puck Recovery'
    )
    
    # Annotate player names if valid data exists
    if valid_data is not None:
        x_col, y_col = get_coordinate_columns(valid_data)
        if x_col and y_col:
            max_annotations = 20
            for i, (_, row) in enumerate(valid_data.iterrows()):
                if i < max_annotations and 'Player' in row and pd.notna(row['Player']):
                    ax.annotate(row['Player'], (row[x_col], row[y_col]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
    
    ax.legend(fontsize=10)
    ax.set_title(f'Puck Recoveries - {selected_team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Recovery statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_recoveries = len(team_recoveries)
        st.metric("Total Recoveries", total_recoveries)
    
    with col2:
        if 'Player' in team_recoveries.columns:
            top_player = team_recoveries['Player'].value_counts().index[0]
            top_player_recoveries = team_recoveries['Player'].value_counts().iloc[0]
            st.metric("Top Recovery Player", f"{top_player} ({top_player_recoveries})")
    
    with col3:
        x_col, _ = get_coordinate_columns(team_recoveries)
        if x_col:
            # Calculate recovery zones
            rink_length = 200
            offensive_zone = len(team_recoveries[team_recoveries[x_col] > rink_length * 0.67])
            st.metric("Offensive Zone Recoveries", offensive_zone)
    
    # Show recovery breakdown by player
    if 'Player' in team_recoveries.columns:
        st.subheader("Recovery Breakdown by Player")
        player_recoveries = team_recoveries['Player'].value_counts()
        
        # Create a proper bar chart with matplotlib for better formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        player_recoveries.plot(kind='bar', ax=ax, color='orange', alpha=0.7)
        plt.title(f'Puck Recoveries by Player - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Player', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Recoveries', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)

def plot_dump_in_out(df):
    """Create dump in/out visualization"""
    st.markdown('<div class="section-header">Dump In / Dump Out</div>', unsafe_allow_html=True)
    
    # Filter for dump events
    dump_events = ['Dump In/Out']
    dump_df = df[df['Event'].isin(dump_events)].copy()
    
    if dump_df.empty:
        st.warning("No dump events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(dump_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="dump_team")
    
    # Filter by selected team
    team_dumps = dump_df[dump_df['Team'] == selected_team]
    
    if team_dumps.empty:
        st.warning(f"No dump data found for {selected_team}.")
        return
    
    # Dump outcome filter
    st.subheader("Dump Outcome Filter")
    if 'Detail 1' in team_dumps.columns:
        # Get available outcomes
        available_outcomes = sorted(team_dumps['Detail 1'].unique())
        
        # Add "All Outcomes" option
        outcome_options = ["All Outcomes"] + available_outcomes
        
        selected_outcome = st.selectbox(
            "Select Dump Outcome:",
            options=outcome_options,
            help="Choose a specific dump outcome to analyze or 'All Outcomes' to see all dump data"
        )
        
        # Filter by selected outcome
        if selected_outcome != "All Outcomes":
            team_dumps = team_dumps[team_dumps['Detail 1'] == selected_outcome]
            if team_dumps.empty:
                st.warning(f"No dump data found for {selected_outcome} outcome.")
                return
    else:
        st.warning("Detail 1 column not found in the dataset.")
        return
    
    # Validate coordinates
    if not validate_coordinates(dump_df, "dump events"):
        st.warning("Cannot create dump map without coordinate data.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Color mapping for dump outcomes
    color_map = {
        'Retained': 'green',
        'Lost': 'red'
    }
    
    # Plot dumps with coordinate validation
    if selected_outcome == "All Outcomes":
        # Plot by outcome (Retained vs Lost)
        for outcome in ['Retained', 'Lost']:
            if outcome in team_dumps['Detail 1'].values:
                outcome_data = team_dumps[team_dumps['Detail 1'] == outcome]
                valid_data = plot_coordinates_on_rink(
                    outcome_data, ax, 
                    color=color_map.get(outcome, 'gray'),
                    size=60, alpha=0.7, label=f'Dump {outcome}'
                )
                
                # Add annotations for key dumps
                if valid_data is not None:
                    x_col, y_col = get_coordinate_columns(valid_data)
                    if x_col and y_col:
                        for i, (_, row) in enumerate(valid_data.iterrows()):
                            if i < 8:  # Limit annotations
                                player = row.get('Player', 'Unknown')
                                ax.annotate(f"{player}\nDump {outcome}", 
                                           (row[x_col], row[y_col]), 
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.8, 
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        # Plot single outcome
        outcome_color = color_map.get(selected_outcome, 'blue')
        valid_data = plot_coordinates_on_rink(
            team_dumps, ax, 
            color=outcome_color, size=70, alpha=0.8, label=f'Dump {selected_outcome}'
        )
        
        # Add annotations for single outcome
        if valid_data is not None:
            x_col, y_col = get_coordinate_columns(valid_data)
            if x_col and y_col:
                for i, (_, row) in enumerate(valid_data.iterrows()):
                    if i < 12:  # More annotations for single outcome
                        player = row.get('Player', 'Unknown')
                        ax.annotate(f"{player}\nDump {selected_outcome}", 
                                   (row[x_col], row[y_col]), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.legend(fontsize=10)
    
    # Update title based on outcome selection
    if selected_outcome == "All Outcomes":
        title = f'Dump In / Dump Out - {selected_team}'
    else:
        title = f'Dump {selected_outcome} - {selected_team}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Dump statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_dumps = len(team_dumps)
        if selected_outcome == "All Outcomes":
            st.metric("Total Dumps", total_dumps)
        else:
            st.metric(f"Dumps {selected_outcome}", total_dumps)
    
    with col2:
        if selected_outcome == "All Outcomes":
            if 'Detail 1' in team_dumps.columns:
                retained = len(team_dumps[team_dumps['Detail 1'] == 'Retained'])
                st.metric("Dumps Retained", retained)
            else:
                st.metric("Dumps Retained", 0)
        else:
            st.metric("Success Rate", "100%" if selected_outcome == "Retained" else "0%")
    
    with col3:
        if selected_outcome == "All Outcomes":
            if 'Detail 1' in team_dumps.columns:
                lost = len(team_dumps[team_dumps['Detail 1'] == 'Lost'])
                st.metric("Dumps Lost", lost)
            else:
                st.metric("Dumps Lost", 0)
        else:
            # Calculate zone distribution for single outcome
            x_col, _ = get_coordinate_columns(team_dumps)
            if x_col:
                offensive_zone = len(team_dumps[team_dumps[x_col] > 133])
                st.metric("Offensive Zone", offensive_zone)
    
    with col4:
        if selected_outcome == "All Outcomes":
            if 'Detail 1' in team_dumps.columns:
                retained = len(team_dumps[team_dumps['Detail 1'] == 'Retained'])
                st.metric("Possession Retained", retained)
        else:
            # Show unique players for single outcome
            unique_players = team_dumps['Player'].nunique()
            st.metric("Active Players", unique_players)
    
    # Show possession details if available
    if 'Detail 1' in team_dumps.columns:
        st.subheader("Possession Details")
        possession_details = team_dumps['Detail 1'].value_counts()
        
        # Create a proper bar chart with matplotlib for better formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        possession_details.plot(kind='bar', ax=ax, color='green', alpha=0.7)
        plt.title(f'Dump Possession Details - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Possession Outcome', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Dumps', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Player information section
    st.subheader("Player Information")
    
    if selected_outcome == "All Outcomes":
        # Show player breakdown for all outcomes
        if 'Player' in team_dumps.columns:
            player_dumps = team_dumps.groupby(['Player', 'Detail 1']).size().reset_index(name='Count')
            player_dumps = player_dumps.pivot(index='Player', columns='Detail 1', values='Count').fillna(0)
            player_dumps['Total'] = player_dumps.sum(axis=1)
            player_dumps = player_dumps.sort_values('Total', ascending=False)
            
            st.write("**Dump Statistics by Player:**")
            st.dataframe(player_dumps, use_container_width=True)
            
            # Show top performers
            col1, col2 = st.columns(2)
            with col1:
                if 'Retained' in player_dumps.columns:
                    top_retained = player_dumps.nlargest(5, 'Retained')[['Retained']]
                    st.write("**Top Players - Dumps Retained:**")
                    st.dataframe(top_retained, use_container_width=True)
            
            with col2:
                if 'Lost' in player_dumps.columns:
                    top_lost = player_dumps.nlargest(5, 'Lost')[['Lost']]
                    st.write("**Top Players - Dumps Lost:**")
                    st.dataframe(top_lost, use_container_width=True)
    else:
        # Show detailed player information for single outcome
        if 'Player' in team_dumps.columns:
            # Player breakdown for selected outcome
            player_counts = team_dumps['Player'].value_counts().reset_index()
            player_counts.columns = ['Player', f'{selected_outcome} Count']
            
            st.write(f"**Players with {selected_outcome} Dumps:**")
            st.dataframe(player_counts, use_container_width=True)
            
            # Show top 5 players
            top_players = player_counts.head(5)
            st.write(f"**Top 5 Players - {selected_outcome} Dumps:**")
            
            for i, (_, row) in enumerate(top_players.iterrows(), 1):
                player = row['Player']
                count = row[f'{selected_outcome} Count']
                st.write(f"{i}. **{player}** - {count} dumps")
            


def plot_zone_entries(df):
    """Create zone entries visualization"""
    st.markdown('<div class="section-header">Zone Entries</div>', unsafe_allow_html=True)
    
    # Filter for zone entry events
    entry_df = df[df['Event'] == 'Zone Entry'].copy()
    
    if entry_df.empty:
        st.warning("No zone entry events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(entry_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="entry_team")
    
    # Filter by selected team
    team_entries = entry_df[entry_df['Team'] == selected_team]
    
    if team_entries.empty:
        st.warning(f"No zone entry data found for {selected_team}.")
        return
    
    # Player selection
    st.subheader("Player Selection")
    
    # Entry Skater (Player 1) selection
    if 'Player' in team_entries.columns:
        entry_skaters = sorted(team_entries['Player'].unique())
        selected_entry_skaters = st.multiselect(
            "Select Entry Skaters (Player 1):",
            options=entry_skaters,
            default=entry_skaters,  # Select all by default
            help="Choose specific entry skaters to analyze. Leave empty to analyze all players."
        )
        
        # Filter by selected entry skaters
        if selected_entry_skaters:
            team_entries = team_entries[team_entries['Player'].isin(selected_entry_skaters)]
        else:
            st.info("No entry skaters selected. Showing all players.")
    
    # Targeted Defender (Player 2) selection
    if 'Player 2' in team_entries.columns:
        targeted_defenders = sorted(team_entries['Player 2'].unique())
        selected_defenders = st.multiselect(
            "Select Targeted Defenders (Player 2):",
            options=targeted_defenders,
            default=targeted_defenders,  # Select all by default
            help="Choose specific targeted defenders to analyze. Leave empty to analyze all defenders."
        )
        
            # Filter by selected defenders
    if selected_defenders:
        team_entries = team_entries[team_entries['Player 2'].isin(selected_defenders)]
    else:
        st.info("No targeted defenders selected. Showing all defenders.")
    
    # Validate coordinates
    if not validate_coordinates(team_entries, "zone entry events"):
        st.warning("Cannot create zone entry map without coordinate data.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Player color mapping for entry skaters
    if 'Player' in team_entries.columns and len(selected_entry_skaters) > 1:
        # Assign different colors to each entry skater
        player_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        player_color_map = {}
        
        for i, player in enumerate(selected_entry_skaters):
            player_color_map[player] = player_colors[i % len(player_colors)]
        
        # Plot zone entries by entry skater
        for player in selected_entry_skaters:
            player_data = team_entries[team_entries['Player'] == player]
            if len(player_data) > 0:
                valid_data = plot_coordinates_on_rink(
                    player_data, ax, 
                    color=player_color_map[player], size=80, alpha=0.8, 
                    label=f'{player} ({len(player_data)} entries)'
                )
                
                # Annotate some entry locations
                if valid_data is not None:
                    x_col, y_col = get_coordinate_columns(valid_data)
                    if x_col and y_col:
                        for i, (_, row) in enumerate(valid_data.iterrows()):
                            if i < 5:  # Limit annotations per player
                                entry_type = row.get('Detail 1', 'Entry')
                                defender = row.get('Player 2', 'Unknown')
                                ax.annotate(f"{row['Player']}\n{entry_type}\nvs {defender}", 
                                           (row[x_col], row[y_col]), 
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.8, 
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        # Single player or entry type analysis
        team_colors = {'Team A': 'red', 'Team B': 'green', 'Team C': 'blue', 'Team D': 'orange'}
        team_color = team_colors.get(selected_team, 'purple')
        
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
                if len(entry_data) > 0:
                    valid_data = plot_coordinates_on_rink(
                        entry_data, ax, 
                        color=[color_map[entry_type]], size=80, alpha=0.8, 
                        label=f'{entry_type} ({len(entry_data)})'
                    )
                    
                    # Annotate some entry locations
                    if valid_data is not None:
                        x_col, y_col = get_coordinate_columns(valid_data)
                        if x_col and y_col:
                            for i, (_, row) in enumerate(valid_data.iterrows()):
                                if i < 8:  # Limit annotations
                                    player = row.get('Player', 'Unknown')
                                    defender = row.get('Player 2', 'Unknown')
                                    ax.annotate(f"{player}\n{entry_type}\nvs {defender}", 
                                               (row[x_col], row[y_col]), 
                                               xytext=(5, 5), textcoords='offset points',
                                               fontsize=8, alpha=0.8, 
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        else:
            # If no entry types, just plot all entries with team color
            valid_data = plot_coordinates_on_rink(
                team_entries, ax, 
                color=team_color, size=80, alpha=0.8, label=f'{selected_team} Zone Entries'
            )
            
            # Annotate some entry locations
            if valid_data is not None:
                x_col, y_col = get_coordinate_columns(valid_data)
                if x_col and y_col:
                    for i, (_, row) in enumerate(valid_data.iterrows()):
                        if i < 10:  # Limit annotations
                            player = row.get('Player', 'Unknown')
                            defender = row.get('Player 2', 'Unknown')
                            ax.annotate(f"{player}\nvs {defender}", 
                                       (row[x_col], row[y_col]), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, alpha=0.8)
    
    ax.legend(fontsize=10)
    
    # Update title based on player selection
    if len(selected_entry_skaters) == 1:
        title = f'Zone Entries - {selected_entry_skaters[0]} ({selected_team})'
    elif len(selected_entry_skaters) < len(entry_skaters):
        title = f'Zone Entries - Selected Players ({selected_team})'
    else:
        title = f'Zone Entries - {selected_team}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Entry statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_entries = len(team_entries)
        st.metric("Total Zone Entries", total_entries)
    
    with col2:
        if 'Player' in team_entries.columns and len(team_entries) > 0:
            top_player = team_entries['Player'].value_counts().index[0]
            top_player_entries = team_entries['Player'].value_counts().iloc[0]
            st.metric("Top Entry Player", f"{top_player} ({top_player_entries})")
    
    with col3:
        if 'Detail 1' in team_entries.columns:
            carried_entries = len(team_entries[team_entries['Detail 1'] == 'Carried'])
            st.metric("Carried Entries", carried_entries)
    
    # Show entry type breakdown
    if 'Detail 1' in team_entries.columns and len(team_entries) > 0:
        st.subheader("Entry Type Breakdown")
        entry_types = team_entries['Detail 1'].value_counts()
        
        # Create a proper bar chart with matplotlib for better formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        entry_types.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
        plt.title(f'Zone Entry Types - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Entry Type', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Entries', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Show detailed entry breakdown
        st.subheader("Detailed Zone Entry Breakdown")
        entry_details = team_entries.groupby(['Player', 'Detail 1']).size().reset_index(name='Entries')
        entry_details = entry_details.sort_values('Entries', ascending=False)
        st.dataframe(entry_details, use_container_width=True)
        
        # Show defender matchup analysis
        if 'Player 2' in team_entries.columns and len(team_entries) > 0:
            st.subheader("Defender Matchup Analysis")
            defender_analysis = team_entries.groupby(['Player 2']).size().reset_index(name='Entries Against')
            defender_analysis = defender_analysis.sort_values('Entries Against', ascending=False)
            st.dataframe(defender_analysis, use_container_width=True)
    
    # Player information section
    st.subheader("Player Information")
    
    if 'Player' in team_entries.columns:
        # Player breakdown
        player_entries = team_entries['Player'].value_counts().reset_index()
        player_entries.columns = ['Player', 'Zone Entries']
        
        st.write("**Zone Entries by Player:**")
        st.dataframe(player_entries, use_container_width=True)
        
        # Show top 5 players
        top_players = player_entries.head(5)
        st.write("**Top 5 Players - Zone Entries:**")
        
        for i, (_, row) in enumerate(top_players.iterrows(), 1):
            player = row['Player']
            entries = row['Zone Entries']
            st.write(f"{i}. **{player}** - {entries} entries")
        
        # Entry type breakdown by player
        if 'Detail 1' in team_entries.columns:
            st.write("**Entry Types by Player:**")
            entry_type_player = team_entries.groupby(['Player', 'Detail 1']).size().reset_index(name='Count')
            entry_type_player = entry_type_player.pivot(index='Player', columns='Detail 1', values='Count').fillna(0)
            entry_type_player = entry_type_player.sort_values(entry_type_player.columns[0], ascending=False)
            st.dataframe(entry_type_player, use_container_width=True)

def plot_faceoff_wins(df):
    """Create faceoff wins visualization"""
    st.markdown('<div class="section-header">Faceoff Wins</div>', unsafe_allow_html=True)
    
    # Filter for faceoff events
    faceoff_df = df[df['Event'] == 'Faceoff Win'].copy()
    
    if faceoff_df.empty:
        st.warning("No faceoff win events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(faceoff_df['Team'].unique())
    selected_team = st.selectbox("Select Team:", teams, key="faceoff_team")
    
    # Filter by selected team
    team_faceoffs = faceoff_df[faceoff_df['Team'] == selected_team]
    
    if team_faceoffs.empty:
        st.warning(f"No faceoff data found for {selected_team}.")
        return
    
    # Player selection
    if 'Player' in team_faceoffs.columns:
        st.subheader("Player Selection")
        
        # Get all players who won faceoffs
        faceoff_players = sorted(team_faceoffs['Player'].unique())
        
        # Multi-select for players
        selected_players = st.multiselect(
            "Select Players to Analyze:",
            options=faceoff_players,
            default=faceoff_players,  # Select all by default
            help="Choose specific players to analyze. Leave empty to analyze all players."
        )
        
        # Filter by selected players
        if selected_players:
            team_faceoffs = team_faceoffs[team_faceoffs['Player'].isin(selected_players)]
        else:
            st.info("No players selected. Showing all players.")
    
    # Validate coordinates
    if not validate_coordinates(team_faceoffs, "faceoff events"):
        st.warning("Cannot create faceoff map without coordinate data.")
        return
    
    # Create rink plot
    fig, ax = create_rink_plot()
    
    # Player color mapping
    if 'Player' in team_faceoffs.columns and len(selected_players) > 1:
        # Assign different colors to each player
        player_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        player_color_map = {}
        
        for i, player in enumerate(selected_players):
            player_color_map[player] = player_colors[i % len(player_colors)]
        
        # Plot faceoff locations by player
        for player in selected_players:
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
        team_color = team_colors.get(selected_team, 'purple')
        
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
    ax.set_title(f'Faceoff Wins - {selected_team}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
    
    # Ensure proper axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 85)
    
    st.pyplot(fig)
    
    # Faceoff statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_faceoffs = len(team_faceoffs)
        st.metric("Total Faceoff Wins", total_faceoffs)
    
    with col2:
        if 'Player' in team_faceoffs.columns and len(team_faceoffs) > 0:
            top_player = team_faceoffs['Player'].value_counts().index[0]
            top_player_wins = team_faceoffs['Player'].value_counts().iloc[0]
            st.metric("Top Faceoff Winner", f"{top_player} ({top_player_wins})")
    
    with col3:
        x_col, _ = get_coordinate_columns(team_faceoffs)
        if x_col:
            # Calculate faceoff zones
            rink_length = 200
            offensive_zone = len(team_faceoffs[team_faceoffs[x_col] > rink_length * 0.67])
            st.metric("Offensive Zone Wins", offensive_zone)
    
    # Show faceoff winners breakdown
    if 'Player' in team_faceoffs.columns and len(team_faceoffs) > 0:
        st.subheader("Faceoff Winners Breakdown")
        player_faceoffs = team_faceoffs['Player'].value_counts()
        
        # Create a proper bar chart with matplotlib for better formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        player_faceoffs.plot(kind='bar', ax=ax, color='red', alpha=0.7)
        plt.title(f'Faceoff Wins by Player - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Player', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Faceoff Wins', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Show detailed faceoff breakdown
        st.subheader("Detailed Faceoff Breakdown")
        faceoff_details = team_faceoffs.groupby(['Player']).size().reset_index(name='Wins')
        faceoff_details = faceoff_details.sort_values('Wins', ascending=False)
        st.dataframe(faceoff_details, use_container_width=True)

def plot_penalties(df):
    """Create penalties visualization"""
    st.markdown('<div class="section-header">Penalties</div>', unsafe_allow_html=True)
    
    # Filter for penalty events
    penalty_df = df[df['Event'] == 'Penalty Taken'].copy()
    
    if penalty_df.empty:
        st.warning("No penalty events found in the dataset.")
        return
    
    # Team selection
    teams = sorted(penalty_df['Team'].unique())
    
    # Add team comparison option
    compare_teams = st.checkbox("Compare Multiple Teams", key="compare_teams")
    
    if compare_teams:
        selected_teams = st.multiselect(
            "Select Teams to Compare:",
            options=teams,
            default=teams[:2] if len(teams) >= 2 else teams,
            help="Select multiple teams to compare penalty patterns"
        )
        
        if not selected_teams:
            st.warning("Please select at least one team for comparison.")
            return
        
        # Filter by selected teams
        team_penalties = penalty_df[penalty_df['Team'].isin(selected_teams)]
        selected_team = selected_teams[0]  # For display purposes
    else:
        selected_team = st.selectbox("Select Team:", teams, key="penalty_team")
        selected_teams = [selected_team]
        
        # Filter by selected team
        team_penalties = penalty_df[penalty_df['Team'] == selected_team]
    
    if team_penalties.empty:
        st.warning(f"No penalty data found for {selected_team}.")
        return
    
    # Player selection
    if 'Player' in team_penalties.columns:
        st.subheader("Player Selection")
        
        # Get all players who took penalties
        penalty_players = sorted(team_penalties['Player'].unique())
        
        # Multi-select for players
        selected_players = st.multiselect(
            "Select Players to Analyze:",
            options=penalty_players,
            default=penalty_players,  # Select all by default
            help="Choose specific players to analyze. Leave empty to analyze all players."
        )
        
        # Filter by selected players
        if selected_players:
            team_penalties = team_penalties[team_penalties['Player'].isin(selected_players)]
        else:
            st.info("No players selected. Showing all players.")
    
    # Penalty statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_penalties = len(team_penalties)
        st.metric("Total Penalties", total_penalties)
    
    with col2:
        if 'Player' in team_penalties.columns and len(team_penalties) > 0:
            top_player = team_penalties['Player'].value_counts().index[0]
            top_player_penalties = team_penalties['Player'].value_counts().iloc[0]
            st.metric("Most Penalized Player", f"{top_player} ({top_player_penalties})")
    
    with col3:
        if 'Detail 1' in team_penalties.columns and len(team_penalties) > 0:
            most_common = team_penalties['Detail 1'].value_counts().index[0]
            st.metric("Most Common Infraction", most_common)
    
    # Show penalty breakdown by infraction type
    if 'Detail 1' in team_penalties.columns and len(team_penalties) > 0:
        st.subheader("Penalties by Infraction Type")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        penalty_types = team_penalties['Detail 1'].value_counts()
        penalty_types.plot(kind='bar', ax=ax, color='red', alpha=0.7)
        plt.title(f'Penalties by Type - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Infraction Type', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Penalties', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Show penalty breakdown by player
    if 'Player' in team_penalties.columns and len(team_penalties) > 0:
        st.subheader("Penalties by Player")
        player_penalties = team_penalties['Player'].value_counts()
        
        # Create a proper bar chart with matplotlib for better formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        player_penalties.plot(kind='bar', ax=ax, color='red', alpha=0.7)
        plt.title(f'Penalties by Player - {selected_team}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Player', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Penalties', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
        
            # Show detailed penalty breakdown
    st.subheader("Detailed Penalty Breakdown")
    penalty_details = team_penalties.groupby(['Player', 'Detail 1']).size().reset_index(name='Count')
    penalty_details = penalty_details.sort_values('Count', ascending=False)
    st.dataframe(penalty_details, use_container_width=True)
    
    # Penalty location analysis
    if 'X' in team_penalties.columns and 'Y' in team_penalties.columns:
        st.subheader("Penalty Locations")
        
        # Create rink plot for penalty locations
        fig, ax = create_rink_plot()
        
        # Team color mapping
        team_colors = {'Team A': 'red', 'Team B': 'green', 'Team C': 'blue', 'Team D': 'orange'}
        
        if compare_teams:
            # Plot multiple teams with different colors
            for team in selected_teams:
                team_data = team_penalties[team_penalties['Team'] == team]
                team_color = team_colors.get(team, 'purple')
                
                if 'Detail 1' in team_data.columns:
                    # Plot by infraction type for each team
                    infraction_types = team_data['Detail 1'].unique()
                    for infraction_type in infraction_types:
                        infraction_data = team_data[team_data['Detail 1'] == infraction_type]
                        if len(infraction_data) > 0:
                            ax.scatter(infraction_data['X'], infraction_data['Y'], 
                                      c=team_color, s=80, alpha=0.8, 
                                      label=f'{team} - {infraction_type} ({len(infraction_data)})')
                            
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
                              c=team_color, s=80, alpha=0.8, label=f'{team} Penalties ({len(team_data)})')
        else:
            # Single team analysis with individual player colors
            if 'Player' in team_penalties.columns and len(selected_players) > 1:
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
                team_color = team_colors.get(selected_team, 'purple')
                
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
                              c=team_color, s=80, alpha=0.8, label=f'{selected_team} Penalties')
                    
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
            ax.set_title(f'Penalty Locations - {selected_team}', fontsize=16, fontweight='bold', pad=20)
            
        ax.set_xlabel('Rink Length (feet)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rink Width (feet)', fontsize=12, fontweight='bold')
        
        # Ensure proper axis limits
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 85)
        
        st.pyplot(fig)
        
        # Zone analysis
        st.subheader("Penalty Zone Analysis")
        
        if compare_teams:
            # Show zone analysis for each team
            for team in selected_teams:
                team_data = team_penalties[team_penalties['Team'] == team]
                team_color = team_colors.get(team, 'purple')
                
                st.write(f"**{team}**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    offensive_zone = len(team_data[team_data['X'] > 133])
                    st.metric("Offensive Zone", offensive_zone)
                
                with col2:
                    neutral_zone = len(team_data[(team_data['X'] >= 67) & (team_data['X'] <= 133)])
                    st.metric("Neutral Zone", neutral_zone)
                
                with col3:
                    defensive_zone = len(team_data[team_data['X'] < 67])
                    st.metric("Defensive Zone", defensive_zone)
        else:
            # Single team zone analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col, _ = get_coordinate_columns(team_penalties)
                if x_col:
                    offensive_zone = len(team_penalties[team_penalties[x_col] > 133])  # 2/3 of rink
                    st.metric("Offensive Zone Penalties", offensive_zone)
            
            with col2:
                x_col, _ = get_coordinate_columns(team_penalties)
                if x_col:
                    neutral_zone = len(team_penalties[(team_penalties[x_col] >= 67) & (team_penalties[x_col] <= 133)])
                    st.metric("Neutral Zone Penalties", neutral_zone)
            
            with col3:
                x_col, _ = get_coordinate_columns(team_penalties)
                if x_col:
                    defensive_zone = len(team_penalties[team_penalties[x_col] < 67])
                    st.metric("Defensive Zone Penalties", defensive_zone)
    
    # Comprehensive penalty data table with sorting
    st.subheader("Complete Penalty Data")
    
    # Create comprehensive dataframe with all relevant information
    penalty_data = team_penalties.copy()
    
    # Add zone information
    x_col, _ = get_coordinate_columns(penalty_data)
    if x_col:
        penalty_data['Zone'] = penalty_data[x_col].apply(
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
    x_col, y_col = get_coordinate_columns(penalty_data)
    if x_col and y_col:
        display_columns.extend([x_col, y_col])
    
    # Filter to only columns that exist
    display_columns = [col for col in display_columns if col in penalty_data.columns]
    
    # Create the display dataframe
    display_df = penalty_data[display_columns].copy()
    
    # Add sorting options
    st.write("**Sort by:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox("Primary Sort:", ['Team', 'Player', 'Detail 1', 'Zone', 'Period'], key="sort1")
    
    with col2:
        sort_order = st.selectbox("Order:", ['Ascending', 'Descending'], key="order1")
    
    with col3:
        secondary_sort = st.selectbox("Secondary Sort:", ['None', 'Team', 'Player', 'Detail 1', 'Zone', 'Period'], key="sort2")
    
    # Apply sorting
    if secondary_sort != 'None' and secondary_sort in display_df.columns:
        display_df = display_df.sort_values([sort_by, secondary_sort], 
                                          ascending=[sort_order == 'Ascending', True])
    else:
        display_df = display_df.sort_values(sort_by, ascending=sort_order == 'Ascending')
    
    # Display the sorted dataframe
    st.dataframe(display_df, use_container_width=True)
    
    # Download option
    csv = display_df.to_csv(index=False)
    if compare_teams:
        filename = f"team_comparison_penalties.csv"
    else:
        filename = f"{selected_team}_penalties.csv"
    
    st.download_button(
        label="Download Penalty Data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def main():
    # Header
    st.markdown('<div class="main-header">Women\'s Hockey Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your hockey data CSV file. If no file is uploaded, the app will load 'olympic_womens_dataset.csv'."
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("Please upload a CSV file or ensure 'olympic_womens_dataset.csv' is in the working directory.")
        st.stop()
    
    # Display basic dataset info
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"**Rows:** {len(df)}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")
    
    if 'Team' in df.columns:
        teams = sorted(df['Team'].unique())
        st.sidebar.write(f"**Teams:** {len(teams)}")
        st.sidebar.write(", ".join(teams[:3]) + ("..." if len(teams) > 3 else ""))
    
    if 'Event' in df.columns:
        events = df['Event'].value_counts()
        st.sidebar.write("**Top Events:**")
        for event, count in events.head(5).items():
            st.sidebar.write(f"• {event}: {count}")
    
    # Coordinate information
    x_col, y_col = get_coordinate_columns(df)
    if x_col and y_col:
        st.sidebar.header("Coordinate System")
        st.sidebar.write("**Rink Dimensions:** 200' x 85'")
        st.sidebar.write("**X-axis:** Rink length (0-200 feet)")
        st.sidebar.write("**Y-axis:** Rink width (0-85 feet)")
        
        # Show coordinate ranges
        x_range = f"{df[x_col].min():.0f} - {df[x_col].max():.0f}"
        y_range = f"{df[y_col].min():.0f} - {df[y_col].max():.0f}"
        st.sidebar.write(f"**X Range:** {x_range}")
        st.sidebar.write(f"**Y Range:** {y_range}")
        
        # Zone information
        st.sidebar.write("**Zones:**")
        st.sidebar.write("• Defensive: X < 67")
        st.sidebar.write("• Neutral: 67 ≤ X ≤ 133")
        st.sidebar.write("• Offensive: X > 133")
    else:
        st.sidebar.header("Coordinate System")
        st.sidebar.warning("No X,Y coordinates found in dataset")
    
    # Navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Choose Visualization:",
        ["All Events Map", "Shot & Goal Map", "Passing Network", "Takeaways", "Puck Recoveries", 
         "Dump In / Dump Out", "Zone Entries", "Faceoff Wins", "Penalties"]
    )
    
    # Display selected data preview
    st.sidebar.header("Data Preview")
    if st.sidebar.checkbox("Show data preview"):
        st.sidebar.dataframe(df.head(10), use_container_width=True)
    
    # Main content area
    st.markdown("---")
    
    # Routing logic
    if option == "All Events Map":
        plot_all_events_map(df)
    elif option == "Shot & Goal Map":
        plot_shot_goal_map(df)
    elif option == "Passing Network":
        plot_passing_network(df)
    elif option == "Takeaways":
        plot_takeaways(df)
    elif option == "Puck Recoveries":
        plot_puck_recoveries(df)
    elif option == "Dump In / Dump Out":
        plot_dump_in_out(df)
    elif option == "Zone Entries":
        plot_zone_entries(df)
    elif option == "Faceoff Wins":
        plot_faceoff_wins(df)
    elif option == "Penalties":
        plot_penalties(df)

if __name__ == "__main__":
    main()
