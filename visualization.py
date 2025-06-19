import matplotlib
matplotlib.use('Agg')  # Headless backend (no GUI required)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

def generate_all_visualizations(df):
    images = []

    def save_current_plot():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{image_base64}"

    # 1. Correlation Matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.4f', annot_kws={'fontsize': 8})
    plt.title("Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    images.append(save_current_plot())

    # 2. Average Conversion Rate by Campaign Type
    avg_by_campaign = df.groupby('Campaign_Type')['Conversion_Rate'].mean()
    _plot_bar(avg_by_campaign, 'Campaign Types', 'Average Conversion Rate (%) by Campaign Type')
    images.append(save_current_plot())

    # 3. Conversion Heatmap by Audience, Campaign Type, Channel
    heatmap_data = df.groupby(['Target_Audience', 'Campaign_Type', 'Channel_Used'])['Conversion_Rate'].mean().unstack()
    plt.figure(figsize=(18, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.2%', linewidths=0.5, cbar_kws={'label': 'Conversion Rate'}, annot_kws={'fontsize': 7})
    plt.title('Conversion Rate by Target Audience, Campaign Type, and Channel')
    plt.xlabel('Channel')
    plt.ylabel('Target Audience / Campaign Type')
    plt.tight_layout()
    images.append(save_current_plot())

    # 4–8. Bar Plots
    avg_by_segment = df.groupby('Customer_Segment')['Conversion_Rate'].mean()
    images.append(_plot_bar(avg_by_segment, 'Customer Segment', 'Average Conversion Rate (%) by Customer Segment', return_img=True))

    avg_by_channel = df.groupby('Channel_Used')['Conversion_Rate'].mean()
    images.append(_plot_bar(avg_by_channel, 'Channel', 'Average Conversion Rate (%) by Channel', return_img=True))

    avg_by_duration = df.groupby('Duration')['Conversion_Rate'].mean()
    images.append(_plot_bar(avg_by_duration, 'Duration', 'Average Conversion Rate (%) by Duration', return_img=True))

    avg_by_location = df.groupby('Location')['Conversion_Rate'].mean()
    images.append(_plot_bar(avg_by_location, 'Location', 'Average Conversion Rate (%) by Location', return_img=True))

    avg_by_language = df.groupby('Language')['Conversion_Rate'].mean()
    images.append(_plot_bar(avg_by_language, 'Language', 'Average Conversion Rate (%) by Language', return_img=True))

    # 9. Correlation: Engagement, ROI, Cost
    selected = df[['Engagement_Score', 'Conversion_Rate', 'Acquisition_Cost', 'ROI']]
    plt.figure(figsize=(14, 6))
    sns.heatmap(selected.corr(), annot=True, cmap='coolwarm', fmt='.4f', annot_kws={'fontsize': 8})
    plt.title('Correlation between Engagement Score and Other Metrics')
    plt.xlabel("Metrics")
    plt.ylabel("Metrics")
    plt.tight_layout()
    images.append(save_current_plot())

    # 10. Time Series: Monthly Conversion Rate
    monthly = df.groupby('Campaign_Month')[['Conversion_Rate']].mean().reset_index()
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly, x='Campaign_Month', y='Conversion_Rate', marker='o')
    plt.title('Average Conversion Rate (%) by Campaign Month')
    plt.xlabel('Month (1 = Jan, ..., 12 = Dec)')
    plt.ylabel('Conversion Rate (%)')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    images.append(save_current_plot())

    return images

def _plot_bar(series, xlabel, title, color_palette=None, return_img=False):
    colors = color_palette if color_palette else ['#FC4100', '#0056B3', '#FFC55A', '#9BC4F8', '#59D5E0', '#FB773C']
    bars = series.plot(kind='bar', color=colors[:len(series)], figsize=(16, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average Conversion Rate (%)")
    plt.xticks(rotation=0)
    for bar in bars.patches:
        height = bar.get_height()
        bars.annotate(f'{height:.2%}', (bar.get_x() + bar.get_width() / 2, height),
                      ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    
    if return_img:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{image_base64}"
