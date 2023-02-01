curl -X POST --location "https://api.twitter.com/2/tweets/search/stream/rules?dry_run=false" \
    -H "Authorization: Bearer $bearer" \
    -H "Content-Type: application/json" \
    -d "{
          \"add\": [
            {
              \"value\": \"#news lang:en\",
              \"tag\": \"english news\"
            }
          ]
        }"

# With delete option

#curl -X POST --location "https://api.twitter.com/2/tweets/search/stream/rules?dry_run=false" \
#    -H "Authorization: Bearer $bearer" \
#    -H "Content-Type: application/json" \
#    -d "{
#          \"add\": [
#            {
#              \"value\": \"#news lang:en\",
#              \"tag\": \"english news\"
#            }
#          ],
#          \"delete\": {
#            \"ids\": []
#          }
#        }"