
const formToJSON = elements => [].reduce.call(elements, (data, element) => {
  data[element.name] = element.value;
  return data;
}, {});


document.addEventListener('DOMContentLoaded', init, false);
function init(){
  function message () {
    console.log('hi')
  }
  var button = document.getElementById('butt');
  button.addEventListener('click', SendAjax, true);
};


function SendAjax(){
    var form = document.querySelector("form");
    const data = formToJSON(form.elements);
    //console.log(data)
    //console.log(JSON.stringify(data, null, "  "));

    $.post('/hist',
        { data: data,
        processDataBoolean: false },
        function(data2, status, xhr) {
          console.log(data2/(data['bill']))
            if (data2/(data['bill'])<=0.14){
              temp=Math.round(data['bill']*0.14)
              document.getElementById("tip").innerHTML='Suggested total tip amount: '+temp;
            }else{
              document.getElementById("tip").innerHTML='Suggested total tip amount: '+data2;
            }
            form.reset()

            //alert('status: ' + status + ', data: ' + data);

        }).done(function() {  })
        .fail(function(jqxhr, settings, ex) { alert('failed, ' + ex); });
}
