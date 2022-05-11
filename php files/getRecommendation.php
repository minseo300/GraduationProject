<?php
    
    // 앱에 사용자 옷으로 매칭한 추천 스타일링 보내주기
    include('connection.php');

    $count=$_GET["COUNT"];
    //echo "count: ".$count."<br/>"; // 추천된 스타일링 개수
    $case=$_GET["CASE"];
    //echo "case: ".$case."<br/>"; // outer-top-bottom or top-bottom
    $values_array=array();
    for($i=0;$i<$count;$i=$i+1)
    {
        if($case=="2")
        {
           
            // $par=$i."TOP";
            // $top=$_GET[$par];
            // $par=$i."BOTTOM";
            // $bottom=$_GET[$par];
            // array_push($values_array,array($top,$bottom));
            array_push($values_array,array($_GET[$i."TOP"],$_GET[$i."BOTTOM"]));
        }
        else if($case=="3")
        {
            // $par=$i."OUTER";
            // $outer=$_GET[$par];
            // $par=$i."TOP";
            // $top=$_GET[$par];
            // $par=$i."BOTTOM";
            // $bottom=$_GET[$par];
            array_push($values_array,array($_GET[$i."OUTER"],$_GET[$i."TOP"],$_GET[$i."BOTTOM"]));
        }
        
    }
    // print_r($values_array);

    $json_result=array();
    for($i=0;$i<$count;$i++)
    {
        $case_=$values_array[$i];
        // print_r($case_);
        if($case=="2")
        {
            $top=$case_[0];
            $bottom=$case_[1];
            
            $sql="SELECT * FROM U_top_image WHERE ID=$top";
            $top_result=mysqli_query($conn,$sql);
            while($row=mysqli_fetch_array($top_result)){
                $top_image=$row['image'];
            }
            //echo "<br/>top_result: ".$top_image;
            //echo "<br/>top image: ".$top_result['image'];
            $sql="SELECT * FROM U_bottom_image WHERE ID=$bottom";
            $botom_result=mysqli_query($conn,$sql);
            while($row=mysqli_fetch_array($botom_result)){
                $bottom_image=$row['image'];
            }
            array_push($json_result,array("Top"=>$top_image,"Top_ID"=>$top,"Bottom"=>$bottom_image,"Bottom_ID"=>$bottom));
            
        }
        else if($case=="3")
        {
            $outer=$case_[0];
            $top=$case_[1];
            $bottom=$case_[2];

            $sql="SELECT * FROM U_top_image WHERE ID=$outer";
            $outer_result=mysqli_query($conn,$sql);
            while($row=mysqli_fetch_array($outer_result)){
                $outer_image=$row['image'];
            }
            $sql="SELECT * FROM U_top_image WHERE ID=$top";
            $top_result=mysqli_query($conn,$sql);
            while($row=mysqli_fetch_array($top_result)){
                $top_image=$row['image'];
            }
            $sql="SELECT * FROM U_bottom_image WHERE ID=$bottom";
            $botom_result=mysqli_query($conn,$sql);
            while($row=mysqli_fetch_array($botom_result)){
                $bottom_image=$row['image'];
            }
            array_push($json_result,array("Outer"=>$outer_image,"Outer_ID"=>$outer,"Top"=>$top_image,"Top_ID"=>$top,"Bottom"=>$bottom_image,"Bottom_ID"=>$bottom));
        }
    }
    
    echo json_encode(array("User_clothes_recommendation"=>$json_result),JSON_UNESCAPED_UNICODE);
    
?>
