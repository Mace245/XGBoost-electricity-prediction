import 'package:flutter/material.dart';

class Management extends StatelessWidget {
  const Management({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body:
        Center(
          child:
            Text("bagas", 
              style: TextStyle(fontSize: 100),
           )
        ),
      )
    );
  }
}