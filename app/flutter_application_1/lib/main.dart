import 'package:flutter/material.dart';
import 'monitor.dart';
import 'predicting.dart';
import 'management.dart';

void main(){
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  int currIndex = 0;
  List<Widget> widgetList = const [
    Monitor(),
    Predicting(),
    Management(),
  ];
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text("Predicting"),
          backgroundColor: Colors.lime,
          titleSpacing: 0,
          centerTitle: true,
        ),
        body: Center(
          child: widgetList[currIndex],
          ),
        bottomNavigationBar: BottomNavigationBar(
          showSelectedLabels: true,
          showUnselectedLabels: true,
          selectedFontSize: 15,
          type: BottomNavigationBarType.shifting,
          currentIndex: currIndex,
          onTap: (index) {
            setState(() {
              currIndex = index;
            });
          },
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.electric_meter),
              label: 'Monitoring',
              backgroundColor: Colors.red,
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.trending_up_rounded),
              label: 'Prediciting',
              backgroundColor: Colors.green,
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.business),
              label: 'Management',
              backgroundColor: Colors.blue,
            ),
          ],
        ),
      ),
    );
  }
}

class Monitoring extends StatelessWidget {
  const Monitoring({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.red,
          title: const Text("Monitoring")
        ),
      )
    );
  }
}