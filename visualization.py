import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def visualize_conversion_insights(df):
    # 1. Avg Conversion Rate by Campaign Type
    avg_by_campaign = df.groupby('Campaign_Type')['Conversion_Rate'].mean()
    _plot_bar(avg_by_campaign, 'Campaign Types', 'Average Conversion Rate (%) by Campaign Type')

    # 2. Heatmap: Conversion by Audience, Campaign Type, Channel
    heatmap_data = df.groupby(['Target_Audience', 'Campaign_Type', 'Channel_Used'])['Conversion_Rate'].mean().unstack()
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2%', linewidths=0.5, cbar_kws={'label': 'Conversion Rate'})
    plt.title('Conversion Rate by Target Audience, Campaign Type, and Channel')
    plt.xlabel('Channel')
    plt.ylabel('Target Audience / Campaign Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 3. Avg Conversion Rate by Customer Segment
    avg_by_segment = df.groupby('Customer_Segment')['Conversion_Rate'].mean()
    _plot_bar(avg_by_segment, 'Customer Segment', 'Average Conversion Rate (%) by Customer Segment')

    # 4. Avg Conversion Rate by Channel
    avg_by_channel = df.groupby('Channel_Used')['Conversion_Rate'].mean()
    _plot_bar(avg_by_channel, 'Channel', 'Average Conversion Rate (%) by Channel')

    # 5. Avg Conversion Rate by Duration
    avg_by_duration = df.groupby('Duration')['Conversion_Rate'].mean()
    _plot_bar(avg_by_duration, 'Duration', 'Average Conversion Rate (%) by Duration')

    # 6. Avg Conversion Rate by Location
    avg_by_location = df.groupby('Location')['Conversion_Rate'].mean()
    _plot_bar(avg_by_location, 'Location', 'Average Conversion Rate (%) by Location')

    # 7. Avg Conversion Rate by Language
    avg_by_language = df.groupby('Language')['Conversion_Rate'].mean()
    _plot_bar(avg_by_language, 'Language', 'Average Conversion Rate (%) by Language')

    # 8. Correlation: Engagement, ROI, Cost
    selected = df[['Engagement_Score', 'Conversion_Rate', 'Acquisition_Cost', 'ROI']]
    plt.figure(figsize=(16, 5))
    sns.heatmap(selected.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Engagement Score and Other Metrics')
    plt.tight_layout()
    plt.show()

    # 9. Time Series: Monthly Conversion Rate
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    monthly = df.groupby(['Year', 'Month'])[['Engagement_Score', 'Conversion_Rate', 'ROI']].mean().reset_index()
    plt.figure(figsize=(16, 5))
    sns.lineplot(data=monthly, x='Month', y='Conversion_Rate', hue='Year', marker='o')
    for line in plt.gca().lines:
        for x_data, y_data in zip(line.get_xdata(), line.get_ydata()):
            plt.text(x_data, y_data, f'{y_data:.2%}', ha='center', va='bottom', color='#0056B3')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Monthly Average Conversion Rate (%) by Year')
    plt.xlabel('Month')
    plt.ylabel('Conversion Rate (%)')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 10. Conversion Rate vs Acquisition Cost Correlation
    conversion_and_cost = df[['Conversion_Rate', 'Acquisition_Cost']].corr()
    plt.figure(figsize=(8, 4))
    sns.heatmap(conversion_and_cost, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Conversion Rate and Acquisition Cost')
    plt.tight_layout()
    plt.show()


def _plot_bar(series, xlabel, title, color_palette=None):
    colors = color_palette if color_palette else ['#FC4100', '#0056B3', '#FFC55A', '#9BC4F8', '#59D5E0', '#FB773C']
    bars = series.plot(kind='bar', color=colors[:len(series)], figsize=(16, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=0)
    plt.ylabel("Average Conversion Rate (%)")
    for bar in bars.patches:
        height = bar.get_height()
        bars.annotate(f'{height:.2%}', (bar.get_x() + bar.get_width() / 2, height),
                      ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.show()
