$(function () {
    $('.tl').click(function () {
        // alert("正在训练中》》》");
        $('.show').empty();
        $('.show').append('正在训练中>>>>>>');
        $.get('/ajax/train/',function (dic) {
            $('.show').empty();
            $('.show').append('训练完毕！耗时'+dic.data);
        })
    });
    $('.ts').click(function () {
        $('.show').empty();
        $('.show').append('正在测试中>>>>>>');
        $.get('/ajax/test/',function (dic) {
            $('.show').empty();
            $('.show').append('测试完毕！正确率'+dic.data);

        })
    });
    $.get('/ajax/all',function (dic) {
        $('.poster-list').empty();
        $.each(dic.data, function(index, item){
             $('.poster-list').append('<li class="poster-item "><a href="#"><img src="static/img/'+item.name+'" width="100%" ><div class="name">'+item.name+'</div></a></li>')
        });
        $('.poster-item').click(function () {
             var name=$(this).text();
        $.get('/ajax/predict/',{"name":name},function (dic) {
            alert("预测类别:"+dic.data+"\n实际类别:"+dic.lable)
        })
        })

    });
    //  $('.poster-item').live('click',function () {
    //     var name=$(this).text();
    //     alert(name)
    //     $.get('/ajax/predict/',{"name":name},function (dic) {
    //         alert("预测类别"+dic.data+"实际类别"+dic.lable)
    //     })
    // });
});