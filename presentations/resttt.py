import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# 1. Load Data
df = pd.read_csv('/Users/jjburrell/Downloads/data-IukTe.csv')

# --- DECLUTTERING STEP ---
# This keeps 70% of the companies randomly to prevent overcrowding
# Change 0.7 to 0.6 or 0.8 to adjust density.
df = df.sample(frac=0.7, random_state=42) 

# 2. Design Settings (The "New York" Aesthetic)
background_color = '#0f0f0f'   # Almost black (Ocean)
land_color = '#1f1f1f'         # Dark Charcoal (Land)
border_color = '#333333'       # Subtle Grey (State Borders)
marker_color = '#00FFC2'       # "Teal/Cyan" Neon
text_color = '#e6e6e6'         # Soft White
font_name = 'sans-serif'       # Clean font

# 3. Create Figure
fig, ax = plt.subplots(figsize=(24, 14)) # Widescreen
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# 4. Load Map Data (Fixed for GeoPandas 1.0)
print("Downloading map geometry...")
map_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(map_url)
usa = world[world['ADMIN'] == "United States of America"]

# 5. Plot the Map
usa.plot(ax=ax, 
         color=land_color, 
         edgecolor=border_color, 
         linewidth=0.5) # Thinner borders = cleaner look

# 6. Plot the "Glow" (Large, transparent dots)
ax.scatter(df['LON'], df['LAT'], 
           color=marker_color, 
           s=200,          # Size of glow
           alpha=0.15,     # Very transparent
           zorder=2,
           edgecolors='none')

# 7. Plot the "Core" (Small, bright dots)
ax.scatter(df['LON'], df['LAT'], 
           color='white', 
           s=15,           # Size of center dot
           alpha=1.0, 
           zorder=3)

# 8. Smart Labeling
texts = []
print("Placing labels...")
for i, row in df.iterrows():
    if pd.notnull(row['LON']) and pd.notnull(row['LAT']):
        t = ax.text(row['LON'], row['LAT'], 
                    f"  {row['Company']}", # Added spaces for padding
                    color=text_color, 
                    fontsize=9, 
                    fontname=font_name,
                    fontweight='normal',
                    zorder=4)
        texts.append(t)

# This pushes text away from dots so they don't overlap
adjust_text(texts, 
            force_text=(0.5, 1.0),    # Push text slightly harder
            force_points=(0.2, 0.5),
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle='-', color='#444444', lw=0.5, alpha=0.6))

# 9. Final Polish
ax.set_title('Under Represented Market in Defense', 
             fontsize=24, 
             color='white', 
             pad=30, 
             fontname=font_name, 
             fontweight='light',
             loc='left') # Align title to left like a dashboard

ax.axis('off') # Turn off the X/Y numbers
ax.set_xlim([-128, -65]) # Focus on Continental US
ax.set_ylim([23, 52])

plt.tight_layout()
plt.savefig('US_Market_Map_Clean2.png', dpi=300, facecolor=background_color, bbox_inches='tight')
print("Done! Map saved as 'US_Market_Map_Clean.png'")
plt.show()