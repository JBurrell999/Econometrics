import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from adjustText import adjust_text  # You may need to run: pip install adjustText

# 1. Load your data
# Make sure 'data-IukTe.csv' is in the same folder
df = pd.read_csv('/Users/jjburrell/Downloads/data-IukTe.csv')

# 2. Setup the "New York" Dark Aesthetic
plt.rcParams['font.family'] = 'sans-serif'
background_color = '#1a1a1a'  # Dark charcoal
land_color = '#2b2b2b'        # Slightly lighter grey
border_color = '#444444'      # Subtle state borders
accent_color = '#00FFFF'      # Cyan Neon for markers
text_color = '#E0E0E0'        # Off-white for text

fig, ax = plt.subplots(figsize=(24, 16))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# --- THE FIX IS HERE ---
# We load the map directly from the Natural Earth URL since the built-in function is gone.
print("Downloading map data...")
map_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(map_url)

# In this dataset, the country name column is 'ADMIN' instead of 'name'
usa = world[world['ADMIN'] == "United States of America"]
# -----------------------

# 3. Plot the Base Map
usa.plot(ax=ax, 
         color=land_color, 
         edgecolor=border_color, 
         linewidth=0.8)

# 4. Plot the Companies (The Glow Effect)
# First, a larger, semi-transparent dot for the "glow"
ax.scatter(df['LON'], df['LAT'], 
           color=accent_color, 
           s=150, 
           alpha=0.3, 
           zorder=2,
           edgecolors='none')

# Second, a smaller, solid dot for the center
ax.scatter(df['LON'], df['LAT'], 
           color='white', 
           s=30, 
           alpha=1.0, 
           zorder=3)

# 5. Add Labels with Collision Avoidance
texts = []
for i, row in df.iterrows():
    if pd.notnull(row['LON']) and pd.notnull(row['LAT']):
        texts.append(ax.text(row['LON'], row['LAT'], 
                             row['Company'], 
                             color=text_color, 
                             fontsize=10, 
                             fontweight='bold'))

print("Adjusting labels... (this may take a moment)")
adjust_text(texts, 
            only_move={'points':'y', 'text':'y'}, 
            arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5))

# 6. Formatting and Zoom
ax.set_title('US Market Landscape', fontsize=30, color='white', pad=20)
ax.axis('off')

# Zoom to Continental US
ax.set_xlim([-128, -65])
ax.set_ylim([23, 52])

plt.tight_layout()
output_file = 'US_Market_Map_Final.png'
plt.savefig(output_file, dpi=300, facecolor=background_color, bbox_inches='tight')
print(f"Map saved as {output_file}")
plt.show()