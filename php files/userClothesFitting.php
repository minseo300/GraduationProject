<?php
        // 사용자 옷으로 추천 가상피팅
        
        include('connection.php');
        
        $case=$_GET["CASE"];
        $x=$_GET["X"];
        echo "X: ".$x;
        $y=$_GET["Y"];
        echo "Y: ".$y;
        $w=$_GET["w"];
        echo "w: ".$w;
        $h=$_GET["h"];
        echo "h: ".$h;
        $lh=$_GET["lh"];
        echo "lh: ".$lh;

        $json_result=array();

        $top=$_GET["TOP"];
        $bottom=$_GET["BOTTOM"];
        $top_sql="SELECT * FROM U_top_image WHERE ID=$top";
        $result=mysqli_query($conn,$top_sql);
        while($row=mysqli_fetch_array($result)){
           $top=$row['image'];
        }
        //array_push($json_result,"Top"=>$top);
        $bottom_sql="SELECT * FROM U_bottom_image WHERE ID=$bottom";
        $result=mysqli_query($conn,$bottom_sql);
        while($row=mysqli_fetch_array($result)){
           $bottom=$row['image'];
        }
        //array_push($json_result,"Bottom"=>$bottom);
        if($case=="3")
        {
            $outer=$_GET["OUTER"];
            $outer_sql="SELECT * FROM U_outer_image WHERE ID=$outer";
            $result=mysqli_query($conn,$outer_sql);
            while($row=mysqli_fetch_array($result)){
               $outer=$row['image'];
            }
           // array_push($json_result,"Outer"=>$outer);
        }
        //echo json_encode(array("User_fitting"=>$json_result),JSON_UNESCAPED_UNICODE);
       
        $myWidth='<script>document.write(window.innerWidth);</script>';
        //echo "window width: ".$myWidth;
       // echo "<font color=white> Virtual Fitting </font>";

        

        $ux=$x+100;
        $w=$w*5;
        $cropping=$w/2;
        $h=$h*5;
        $ly=$y+$h;
        $lw=$w-40;
        $lh=$lh*5;
        $lx=$ux+8;

        //echo "<img src='".$top."'>";
        //echo "<img src='".$outer."'>";
        ///////////////////////////////////// cropping success
        //echo "<img src='".$top."' style='position: absolute; z-index:1; float:left; margin-left: ".$x."px; margin-top: ".$y."px; width: ".$w."px; height: ".$h."px;' >";
        //echo "<img src='".$all."' style='position:absolute; float:left; margin-left: 5px; margin-top:50px; width: 100px; height:200px;'>";
        echo "<img src='".$outer."' style='position:absolute; z-index:2; float:left; margin-left: ".$ux."px; margin-top: ".$y."px; clip: rect(0,".$cropping."px, ".$h."px, 0); height: ".$h."px; width: ".$w."px;' >";
        echo "<img src='".$top."' style='position:absolute; z-index:1; float:left; margin-left: ".$ux."px; margin-top: ".$y."px; width: ".$w."px; height: ".$h."px;' >";
        echo "<img src='".$bottom."' style='position:absolute; float:left; margin-left: ".$lx."px; margin-top: ".$ly."px; width: ".$lw."px; height: ".$lh."px;' >";
        
        //////////////////////////////////////////////////////////////////
    ?>
