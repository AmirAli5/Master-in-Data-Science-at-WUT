ssh jean-baptiste.soubaras@129.104.253.32 './kafka_2.12-2.6.0/bin/zookeeper-server-start.sh ./kafka_2.12-2.6.0/config/zookeeper.properties &' &
sleep 5
ssh jean-baptiste.soubaras@129.104.253.32 './kafka_2.12-2.6.0/bin/kafka-server-start.sh ./kafka_2.12-2.6.0/config/server.properties' &

