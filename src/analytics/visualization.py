import os
import pandas as pd
import plotly.express as px
from pathlib import Path

def load_data():
    """Load cleaned data from Parquet file"""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_bookings.parquet"
    return pd.read_parquet(data_path)


def plot_cancellation_rates(df: pd.DataFrame):
    cancel_rates = df.groupby('hotel')['is_canceled'].mean().reset_index()
    overall_rate = df['is_canceled'].mean()

    fig = px.bar(
        cancel_rates, x='hotel', y='is_canceled',
        title='<b>Cancellation Rates by Hotel Type</b>',
        labels={'is_canceled': 'Cancellation Rate', 'hotel': ''},
        text_auto='.1%'  # Show percentages on bars
    )

    # Add horizontal reference line
    fig.add_hline(
        y=overall_rate,
        line_dash="dot",
        annotation_text=f"Overall Rate: {overall_rate:.1%}",
        line_color="red"
    )

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.write_html(project_root / "reports" / "cancellation_rates.html")


def plot_geo_distribution(df: pd.DataFrame):
    country_counts = df['country'].value_counts().nlargest(10).reset_index()
    country_counts.columns = ['country', 'bookings']

    fig = px.choropleth(
        country_counts,
        locations='country',
        locationmode='ISO-3',
        color='bookings',
        title='<b>Top 10 Countries by Bookings</b>',
        color_continuous_scale='Viridis',
        hover_data={'country': True, 'bookings': ':,d'}
    )

    fig.update_layout(coloraxis_colorbar_title_text='Bookings')
    fig.write_html(project_root / "reports" / "geo_distribution.html")


def plot_lead_time_distribution(df: pd.DataFrame):
    median_lead = df['lead_time'].median()

    fig = px.histogram(
        df, x='lead_time', nbins=50,
        title='<b>Lead Time Distribution</b>',
        labels={'lead_time': 'Days Before Arrival'},
        opacity=0.8,
        color_discrete_sequence=['#2E86C1']
    )

    # Add median line
    fig.add_vline(
        x=median_lead,
        line_dash="dash",
        annotation_text=f"Median: {median_lead} days",
        line_color="red"
    )

    # Highlight common booking window (0-30 days)
    fig.add_vrect(
        x0=0, x1=30,
        fillcolor="green", opacity=0.1,
        annotation_text="Short-term bookings"
    )

    fig.write_html(project_root / "reports" / "lead_time_histogram.html")


def plot_revenue_trends(df: pd.DataFrame):
    """Monthly revenue trends"""
    df['revenue'] = df['adr'] * df['total_nights']

    # Group and format dates
    monthly_rev = (
        df.groupby(df['arrival_date'].dt.to_period('M'))['revenue'].sum()
        .reset_index()
    )
    monthly_rev['arrival_date'] = monthly_rev['arrival_date'].dt.strftime('%Y-%m')  # Convert Period to string

    # Find peak month
    peak_month = monthly_rev.loc[monthly_rev['revenue'].idxmax()]

    # Create plot
    fig = px.line(
        monthly_rev,
        x='arrival_date',
        y='revenue',
        title='<b>Monthly Revenue Trends</b>',
        labels={'revenue': 'Revenue (€)', 'arrival_date': 'Month'},
        markers=True
    )

    # Add trendline
    fig.add_scatter(
        x=monthly_rev['arrival_date'],
        y=monthly_rev['revenue'].rolling(3).mean(),
        mode='lines',
        line=dict(dash='dot'),
        name='3-Month Moving Avg'
    )

    # Highlight peak month
    fig.add_annotation(
        x=peak_month['arrival_date'],
        y=peak_month['revenue'],
        text=f"Peak: €{peak_month['revenue'] / 1000:.1f}K",
        showarrow=True,
        arrowhead=1,
        ax=-50,
        ay=-40
    )

    fig.write_html(project_root / "reports" / "revenue_trends.html")


def load_data():
    project_root = Path(__file__).parent.parent.parent
    return pd.read_parquet(project_root / "data" / "processed" / "cleaned_bookings.parquet")


def apply_filters(df, date_range=None, filters=None):
    """Filter dataframe based on request parameters"""
    if date_range:
        start = pd.to_datetime(date_range.get('start', '2015-01'))
        end = pd.to_datetime(date_range.get('end', '2017-12'))
        df = df[(df['arrival_date'] >= start) & (df['arrival_date'] <= end)]

    if filters:
        for field, value in filters.items():
            if field in df.columns:
                df = df[df[field] == value]

    return df
def generate_report(metric: str):
    """Generate specific analytics report with filters"""
    df = load_data()
    # Handle missing parameters


    if metric == "revenue":
        return plot_revenue_trends(df)
    elif metric == "cancellation_rate":
        return plot_cancellation_rates(df)
    elif metric == "geographic_distribution":
        return plot_geo_distribution(df)
    elif metric == "lead_time":
        return plot_lead_time_distribution(df)
    else:
        raise ValueError(f"Unknown metric: {metric}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    generate_report()