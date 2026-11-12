| feature | entity | dtype | description | source | leakage_guard |
|---|---|---|---|---|---|
| user_avg_rating | user | float | Mean rating given by the user in the training window | interactions.rating | train only |
| user_rating_std | user | float | Std-dev of the user's ratings (0 if one rating) | interactions.rating | train only |
| user_activity_count | user | int | Number of training interactions | interactions | train only |
| user_positive_ratio | user | float | Share of the user's ratings >= 4 | interactions.rating | train only |
| user_days_since_last_interaction | user | float | Days between the user's last train interaction and REF_DATE | interactions.timestamp | REF_DATE = train/test cut-off |
| user_tenure_days | user | int | Days between signup and REF_DATE | users.signup_date | REF_DATE |
| user_n_categories | user | int | # distinct item categories the user interacted with | interactions x items.category | train only |
| user_preferred_category | user | category | Most frequent category among the user's positive (>=4) ratings | interactions x items.category | train only |
| country / age_group | user | category | Demographics (after cleaning, 'Unknown' when missing) | users | static |
| activity_segment | user | category | Quartile of activity count: light/medium/active/power | interactions | train only |
| item_avg_rating | item | float | Mean rating received in the training window (global mean if cold) | interactions.rating | train only |
| item_bayes_avg_rating | item | float | Shrunk mean: (C*m + sum)/(C + n), C=20 | interactions.rating | train only |
| item_popularity | item | int | # training interactions (0 for cold items) | interactions | train only |
| item_popularity_score | item | float | log1p(count)/log1p(max count) in [0,1] | interactions | train only |
| item_positive_rate | item | float | Share of the item's ratings >= 4 | interactions.rating | train only |
| item_days_since_last_interaction | item | float | Days between the item's last train interaction and REF_DATE | interactions.timestamp | REF_DATE |
| item_category | item | category | Primary genre (first non-generic genre tag) | items.genres | static |
| item_n_genres | item | int | # distinct genre tags | items.genres | static |
| item_top_audience_age_group | item | category | Age group that rates the item most | interactions x users | train only |
| item_num_pages / item_publish_year | item | int | Pages and year (median-imputed per category when missing/invalid) | items | static |
| item_gr_avg_rating / item_gr_log_num_ratings | item | float | Goodreads global rating and log #ratings (external prior) | items | static |
| user_category_affinity | interaction | float | Share of the user's train interactions in the item's category | user_category_share | train only |
| is_preferred_category | interaction | int | 1 if item category == user preferred category | user x item features | train only |
| item_vs_user_avg_gap | interaction | float | item_bayes_avg_rating - user_avg_rating | user x item features | train only |
| item_age_years_at_interaction | interaction | int | Interaction year - publication year | interactions x items | static |
