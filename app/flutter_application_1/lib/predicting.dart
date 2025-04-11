import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class Predicting extends StatefulWidget {
  const Predicting({super.key});

  @override
  State<Predicting> createState() => _PredictingState();
}

class _PredictingState extends State<Predicting> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              const Text(
                "Line Chart 1",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(
                height: 300,
                child: LineChart(
                  LineChartData(
                    lineBarsData: [
                      LineChartBarData(
                        spots: [
                          FlSpot(0, 1),
                          FlSpot(1, 3),
                          FlSpot(2, 2),
                          FlSpot(3, 1.5),
                          FlSpot(4, 2.5),
                        ],
                        isCurved: true,
                        color: Colors.blue,
                        barWidth: 4,
                        isStrokeCapRound: true,
                        belowBarData: BarAreaData(show: false),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                "Line Chart 2",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(
                height: 300,
                child: LineChart(
                  LineChartData(
                    lineBarsData: [
                      LineChartBarData(
                        spots: [
                          FlSpot(0, 2),
                          FlSpot(1, 1),
                          FlSpot(2, 1.5),
                          FlSpot(3, 2.8),
                          FlSpot(4, 3),
                        ],
                        isCurved: true,
                        color: Colors.red,
                        barWidth: 4,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}