# 🏒 Women's Hockey Analytics Dashboard

A comprehensive analytics dashboard for Olympic women's hockey data, built specifically for the Big Data Cup 2021 dataset. Available in **Streamlit web app**, **standalone Python script**, and **smart launcher** versions.

## 🚀 Quick Start

### Option 1: Smart Launcher (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the smart launcher
python run_dashboard.py
```
The launcher automatically detects your environment and recommends the best option.

### Option 2: Streamlit Web App
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

### Option 3: Standalone Python Script (IDE-friendly)
```bash
# Install dependencies
pip install pandas matplotlib seaborn numpy

# Run interactive mode
python hockey_analytics.py

# Or run with specific options
python hockey_analytics.py --file olympic_womens_dataset.csv --team "Team A" --shots
```

## 📁 Files Overview

- **`app.py`** - Streamlit web application (full feature set)
- **`hockey_analytics.py`** - Standalone Python script (core features)
- **`run_dashboard.py`** - Smart launcher that auto-detects best environment
- **`requirements.txt`** - Python dependencies
- **`olympic_womens_dataset.csv`** - Main dataset (Big Data Cup 2021)
- **`sample_data.csv`** - Example dataset for testing
- **`README.md`** - This documentation

## 🎯 Features

### 📊 Visualizations

#### Streamlit App (Full Feature Set)
1. **All Events Map** - Complete event visualization on rink
2. **Shot & Goal Map** - Interactive shot location visualization
3. **Passing Network** - Player-to-player passing heatmap
4. **Takeaways** - Puck takeaway location analysis
5. **Puck Recoveries** - Recovery tracking and analysis
6. **Dump In / Dump Out** - Dump play analysis with possession tracking
7. **Zone Entries** - Offensive zone entry visualization
8. **Faceoff Wins** - Faceoff success mapping
9. **Penalties** - Comprehensive penalty analysis with location mapping, player selection, and detailed sorting

#### Standalone Script (Core Features)
1. **Shot & Goal Map** - Shot location and goal analysis
2. **Passing Network** - Player passing patterns
3. **Takeaways** - Puck takeaway analysis
4. **Zone Entries** - Zone entry visualization
5. **Faceoff Wins** - Faceoff success mapping
6. **Penalties** - Penalty analysis and statistics

### 🛠 Technical Features
- **Triple Mode**: Smart launcher, Streamlit web app, and standalone script
- **CSV Upload**: File upload capability in web app
- **Auto-detection**: Finds default CSV files automatically
- **Professional Plots**: Proper hockey rink dimensions (200x85 feet)
- **Data Export**: Save plots as high-resolution PNG files
- **Interactive CLI**: Command-line interface for IDE users
- **Environment Detection**: Smart launcher chooses best option

## 💻 Usage Options

### 1. Smart Launcher (`run_dashboard.py`)

**Best for**: Getting started quickly, automatic environment detection

```bash
python run_dashboard.py
```

**Features:**
- Automatically detects available environment
- Recommends best option (Streamlit vs standalone)
- Handles missing dependencies gracefully
- User-friendly interface

### 2. Streamlit Web App (`app.py`)

**Best for**: Web-based analysis, sharing with others, interactive exploration

```bash
streamlit run app.py
```

**Features:**
- Web-based interface
- File upload in sidebar
- Real-time filtering
- Interactive team selection
- Responsive design
- **All 9 visualization types**

### 3. Standalone Python Script (`hockey_analytics.py`)

**Best for**: IDE development, automation, command-line usage, when Streamlit isn't available

#### Interactive Mode
```bash
python hockey_analytics.py
```
Shows a menu-driven interface for selecting analyses.

#### Command-Line Mode
```bash
# Analyze specific file
python hockey_analytics.py --file olympic_womens_dataset.csv

# Generate specific plots
python hockey_analytics.py --file data.csv --shots --passing

# Save plots to files
python hockey_analytics.py --file data.csv --team "Team A" --save "analysis_results"

# Analyze specific players for penalties
python hockey_analytics.py --file data.csv --team "Team A" --players "Player 1,Player 2" --penalties

# Launch Streamlit if available
python hockey_analytics.py --streamlit
```

#### Command-Line Options
- `--file PATH` - Specify CSV file to analyze
- `--team NAME` - Select specific team
- `--players "PLAYER1,PLAYER2"` - Select specific players (for penalties analysis)
- `--save PREFIX` - Save plots with filename prefix
- `--shots` - Generate shot map
- `--passing` - Generate passing network
- `--takeaways` - Generate takeaways analysis
- `--zone-entries` - Generate zone entries analysis
- `--faceoffs` - Generate faceoff wins analysis
- `--penalties` - Generate penalties analysis
- `--streamlit` - Launch Streamlit app (if available)

## 📊 Expected Data Format

The dashboard expects a CSV file with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Event` | Type of event | "Shot", "Goal", "Play", "Takeaway" |
| `Team` | Team name | "Team A", "Canada" |
| `Player` | Player name | "Player 1", "Marie-Philip Poulin" |
| `Player 2` | Second player (for passes) | "Player 2", "Sarah Nurse" |
| `X` | X-coordinate (0-200) | 150 |
| `Y` | Y-coordinate (0-85) | 40 |
| `Detail 1` | Additional details | "Retained", "Carried", "Hooking" |

## 🔧 Installation

### For All Features
```bash
pip install -r requirements.txt
```

### For Standalone Script Only
```bash
pip install pandas matplotlib seaborn numpy
```

### For Development
```bash
# Clone or download files
cd HockeyStatsProject

# Install all dependencies
pip install -r requirements.txt

# Test all versions
python run_dashboard.py          # Test smart launcher
streamlit run app.py             # Test web app
python hockey_analytics.py       # Test standalone script
```

## 🎮 Running in Different Environments

### VSCode / PyCharm / Other IDEs
```python
# Run directly in IDE
python hockey_analytics.py

# Or import and use functions
from hockey_analytics import load_data, plot_shot_goal_map
df = load_data("olympic_womens_dataset.csv")
plot_shot_goal_map(df, team="Team A", save_path="output")
```

### Jupyter Notebook
```python
# Import functions
from hockey_analytics import *

# Load and analyze data
df = load_data("olympic_womens_dataset.csv")
plot_shot_goal_map(df)
plt.show()
```

### Command Line / Terminal
```bash
# Interactive mode
python hockey_analytics.py

# Batch processing
python hockey_analytics.py --file data1.csv --shots --save "game1"
python hockey_analytics.py --file data2.csv --shots --save "game2"
```

## 📈 Output Examples

### Console Output (Standalone)
```
📊 Shot & Goal Map Analysis
Available teams: Team A, Team B
Select team (or press Enter for first team): Team A

📈 Shot Statistics for Team A:
   Total Shots: 15
   Goals: 3
   Shot %: 20.0%
   Shots on Net: 12
```

### File Output
When using `--save` option, generates:
- `analysis_results_shot_map.png`
- `analysis_results_passing_network.png`
- `analysis_results_takeaways.png`
- `analysis_results_zone_entries.png`
- `analysis_results_faceoff_wins.png`
- `analysis_results_penalties.png`
- `analysis_results_penalty_locations.png`
- `analysis_results_penalty_details.csv`

## 🛠 Troubleshooting

### Common Issues

**"No data file found"**
- Ensure CSV file is in the working directory
- Check file permissions
- Verify CSV format matches expected columns
- Default file is `olympic_womens_dataset.csv`

**"Streamlit not available"**
- Install with: `pip install streamlit`
- Or use standalone script: `python hockey_analytics.py`
- Or use smart launcher: `python run_dashboard.py`

**"No events found"**
- Check that Event column contains expected values
- Verify data format matches olympic_womens_dataset.csv

**Plots not displaying in IDE**
- Use `plt.show()` after plotting
- Or save to file with `save_path` parameter

### Data Format Tips
- Event names: 'Shot', 'Goal', 'Play', 'Takeaway', 'Puck Recovery', etc.
- Coordinates: Numeric values 0-200 (X), 0-85 (Y)
- Team names: Consistent spelling across dataset
- Player names: Use consistent naming convention

## 🎯 Use Cases

### For Coaches
- Analyze team performance patterns
- Identify player strengths/weaknesses
- Track possession and zone entry success
- Monitor penalty trends

### For Analysts
- Generate reports for stakeholders
- Compare team statistics
- Export high-quality visualizations
- Batch process multiple games

### For Researchers
- Extract insights from Big Data Cup dataset
- Create reproducible analyses
- Generate publication-ready figures
- Integrate with other analysis tools

## 🤝 Contributing

Feel free to enhance the dashboard by:
- Adding new visualization types
- Improving plot formatting
- Adding statistical analysis features
- Enhancing the CLI interface
- Adding data validation
- Expanding the smart launcher functionality

## 📄 License

This project is open source and available under the MIT License.

---

**Ready to analyze some hockey data?** 🏒

Choose your preferred method:
- **Smart Launcher**: `python run_dashboard.py`
- **Web App**: `streamlit run app.py`
- **Standalone**: `python hockey_analytics.py` 