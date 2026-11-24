# AI_Football
An AI football (soccer) match predictor and fantasy league bot

## Project Goal
A platform where LLM's compete with each other on playing fantasy football

## MVP Scope
A simple match prediction platform with Gemma 4 on the English Premiere League

## Future Plans
Adding more models and fantasy football scoring policies to compete with substitutions and transfers available between matches

## APIs Used
 - [API-Sports](https://api-sports.io/) API: Used to access premire league data.
 - Google GenAI: Used to inference the gemma-4-26b-a4b-it LLM.

## Environment Setup

Create `Code/.env` from `Code/.env.example` and add the required API credentials there. Never commit real credential values.

The application loads these variables:

- `GOOGLE_API_KEY`
- `FOOTBALLDATAIO_KEY`
- `API_SPORTS_KEY`
- `FOOTBALL_DATA_KEY`

When the Streamlit app starts, it runs predictions for the next day's Premier
League fixtures and then repeats daily at 06:00 in the host's local timezone
while the app process is running. It fetches the fixtures before prediction.
Every provider registered in `Code/prediction.py` is called. The "Predict
upcoming" button runs those same registered providers on the stored upcoming
fixtures immediately.